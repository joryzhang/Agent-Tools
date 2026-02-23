import sys
import os
import json
import hashlib
import re
import asyncio
import base64
from typing import List, Any, Dict, Optional
import fitz  # PyMuPDF
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# 配置环境
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import get_openai_virtual_model_client

from app.config import settings
from dotenv import load_dotenv

load_dotenv()

# 修复 Windows 编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class RapidPDFParser:
    def __init__(self, use_ocr: bool = True, full_ocr: bool = False):
        print("🔄 初始化 RapidOCR 解析器...")
        self.use_ocr = use_ocr
        self.full_ocr = full_ocr
        self.ocr = RapidOCR() if use_ocr else None

        self.img_dir = settings.IMG_OUTPUT_DIR
        self.cache_dir = settings.DATA_DIR / "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        self.vlm_client = get_openai_virtual_model_client()
        self.semaphore = asyncio.Semaphore(5)

    def parse_pdf(self, pdf_path: str, file_name: str, ocr_threshold: int = 50) -> Dict[str, Any]:
        """
        [主入口] PDF 解析流程控制器
        逻辑现在变成了线性的流水线，非常清晰。
        """
        file_hash = self._get_file_hash(pdf_path)
        cache_path = self.cache_dir / f"{file_hash}_rapid_v2.json"

        # 1. 缓存检查
        if cache_path.exists():
            print(f"♻️ 读取缓存: {file_hash}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"🚀 开始解析: {pdf_path}")
        doc = fitz.open(pdf_path)
        # [修复点 1] 提前获取总页数
        total_pages_count = len(doc)

        # 2. 初始化容器
        context = {
            "all_texts": [],
            "all_tables": [],
            "saved_images": [],
            "markdown_pages": [],  # 仅存储文本和图片的 Markdown，按页索引
            "file_hash": file_hash
        }

        # 表格处理的状态机 (Pending Buffer)
        table_state = {
            "pending": None,  # 暂存正在处理的跨页表格
            "last_header": None  # 上一页的表头
        }

        # 3. 逐页处理循环
        for page_num, page in enumerate(doc):
            print(f"  📄 Page {page_num + 1}/{len(doc)}")

            # --- A. 提取并处理表格 (更新 all_tables 和 table_state) ---
            # 返回 table_bboxes 用于后续擦除
            table_bboxes = self._process_page_tables(page, page_num, table_state, context["all_tables"])

            # --- B. 提取图片 (更新 saved_images) ---
            # 返回图片相关的 Markdown 片段
            img_md_list = self._process_page_images(page, page_num, context["file_hash"], context["saved_images"])

            # --- C. 擦除表格区域 ---
            self._redact_areas(page, table_bboxes)

            # --- D. 提取剩余文本 (更新 all_texts) ---
            # 返回文本相关的 Markdown 片段
            text_md_list = self._process_page_text(page, page_num, ocr_threshold, context["all_texts"])

            # --- E. 组装当前页的非表格 Markdown ---
            # 注意：此时不包含表格，表格最后统一插
            page_content = f"## 第 {page_num + 1} 页\n\n"

            # 合并图片和文本，按垂直位置 y0 排序 (如果需要严格排序，可以在 process 方法里返回带坐标的元组)
            # 这里简单处理：先放文本，图片穿插其中(简化版)，或者直接追加
            # 为了保持整洁，我们把刚才返回的 list 拼起来
            page_content += "\n".join(text_md_list + img_md_list)

            context["markdown_pages"].append(page_content)

        # 4. 循环结束后的收尾
        # 如果还有没提交的表格，强制提交
        if table_state["pending"]:
            self._commit_table(table_state["pending"], context["all_tables"])

        doc.close()

        # 5. [核心步骤] 将表格回填到 Markdown 中
        full_markdown = self._inject_tables_into_markdown(context["markdown_pages"], context["all_tables"])

        # 6. 构造结果并缓存
        result = {
            "markdown": full_markdown,
            "texts": context["all_texts"],
            "tables": context["all_tables"],
            "saved_images": context["saved_images"],
            "metadata": {
                "file_name": file_name,
                "file_hash": file_hash,
                "total_pages": total_pages_count,
                "parser": "RapidOCR+PyMuPDF_Refactored"
            }
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    # ================= 核心逻辑拆分 =================

    def _process_page_tables(self, page, page_num: int, state: Dict, all_tables: List) -> List[Any]:
        """
        处理单页表格：提取、合并逻辑、提交。
        返回表格的 bbox 列表，供后续擦除使用。
        """
        tables = page.find_tables(snap_tolerance=3)
        bboxes = []

        # 如果本页没表格，说明之前的跨页表格肯定断了，提交它
        if not tables and state["pending"]:
            self._commit_table(state["pending"], all_tables)
            state["pending"] = None

        for i, table in enumerate(tables):
            current_data = table.extract()
            if not current_data: continue
            bboxes.append(table.bbox)

            # --- 跨页合并判定逻辑 ---
            is_merged = False
            pending = state["pending"]

            # 判定条件：有Pending + 本页第一个 + 位于顶部 + 列数一致
            if (pending and i == 0 and table.bbox[1] < 150 and
                    len(current_data[0]) == pending["cols_count"]):

                print(f"    🔗 [合并] 检测到跨页表格 (第{page_num + 1}页)")

                first_row = current_data[0]

                # 情况A: 重复表头 -> 丢弃当前表头
                if first_row == state["last_header"]:
                    current_data = current_data[1:]

                # 情况B: 左侧为空 -> 拼接到上一行 (解决断行问题)
                elif str(first_row[0]).strip() == "":
                    last_row = pending["data"][-1]
                    for col_idx, cell_val in enumerate(first_row):
                        if cell_val and str(cell_val).strip():
                            # 拼接逻辑
                            pending["data"][-1][col_idx] = str(last_row[col_idx]) + str(cell_val)
                    current_data = current_data[1:]  # 删掉这一行

                if current_data:
                    pending["data"].extend(current_data)
                is_merged = True

            # --- 新表格处理 ---
            if not is_merged:
                # 先把旧的存了
                if state["pending"]:
                    self._commit_table(state["pending"], all_tables)

                # 开启新的
                norm_data = self.normalize_table_header(current_data)
                state["pending"] = {
                    "data": norm_data,
                    "page_start": page_num + 1,
                    "cols_count": len(norm_data[0])
                }
                state["last_header"] = norm_data[0]

        return bboxes

    def _process_page_images(self, page, page_num: int, file_hash: str, saved_images: List) -> List[str]:
        """提取图片、OCR、保存。返回 Markdown 片段列表。"""
        md_snippets = []
        try:
            image_list = page.get_image_info(xrefs=True)
            for img_idx, img_info in enumerate(image_list):
                if img_info["size"] < 2048: continue  # 忽略小图

                xref = img_info["xref"]
                base_image = page.parent.extract_image(xref)  # 使用 parent (doc) 提取

                img_name = f"{file_hash}_p{page_num + 1}_img{img_idx + 1}.{base_image['ext']}"
                img_path = os.path.join(self.img_dir, img_name)

                with open(img_path, "wb") as f:
                    f.write(base_image["image"])

                # OCR 识别
                ocr_text = ""
                if self.use_ocr:
                    ocr_text = self._ocr_image(base_image["image"])

                saved_images.append({
                    "hash": file_hash,
                    "name": img_name,
                    "path": img_path,
                    "page": page_num + 1,
                    "description": ocr_text
                })

                if ocr_text.strip():
                    md_snippets.append(f"\n**[图片内容]:** {ocr_text.strip()}\n")
        except Exception as e:
            print(f"    ⚠️ 图片处理出错: {e}")

        return md_snippets

    def _process_page_text(self, page, page_num: int, ocr_threshold: int, all_texts: List) -> List[str]:
        """提取文本（含 OCR 补救）。返回 Markdown 片段列表。"""
        md_snippets = []
        # get_text("blocks") 此时已经是擦除表格后的内容了
        blocks = page.get_text("blocks")
        page_text = ""

        for block in blocks:
            # block: (x0, y0, x1, y1, text, ...)
            text = block[4].strip()
            if text:
                page_text += text + "\n"
                md_snippets.append(text)

        # OCR 补救逻辑 (针对扫描件)
        if len(page_text) < ocr_threshold and self.full_ocr:
            print(f"    🔍 文本过少，尝试整页 OCR...")
            pix = page.get_pixmap(dpi=200)
            ocr_result = self._ocr_image(pix.tobytes("png"))
            if len(ocr_result) > len(page_text) + 20:
                md_snippets = [ocr_result]  # 替换掉原来的文本
                page_text = ocr_result

        if page_text.strip():
            all_texts.append({
                "page": page_num + 1,
                "type": "Text",
                "text": page_text.strip()
            })

        return md_snippets

    def _commit_table(self, pending: Dict, all_tables: List):
        """将 Pending 的表格转 Markdown 并存入 all_tables"""
        if not pending or not pending["data"]: return

        md_content = self._table_to_markdown(pending["data"])
        all_tables.append({
            "page": pending["page_start"],
            "content": md_content,
            "format": "markdown"
        })
        print(f"    ✅ 提交表格 (页码: {pending['page_start']}, 行数: {len(pending['data'])})")

    def _inject_tables_into_markdown(self, markdown_pages: List[str], all_tables: List) -> str:
        """
        [最后一步] 将 all_tables 里的表格内容，插回到对应的 markdown_pages 中。
        策略：插在页面末尾。
        """
        for table in all_tables:
            page_idx = table["page"] - 1
            if 0 <= page_idx < len(markdown_pages):
                # 追加到该页末尾
                markdown_pages[page_idx] += f"\n\n{table['content']}\n\n"

        return "\n\n".join(markdown_pages)

    def _redact_areas(self, page, bboxes):
        """擦除页面上的指定区域"""
        if not bboxes: return
        for bbox in bboxes:
            page.add_redact_annot(bbox)
        page.apply_redactions()

    # ================= 工具函数 =================

    def _ocr_image(self, img_bytes: bytes) -> str:
        if not self.ocr: return ""
        try:
            result, _ = self.ocr(img_bytes)
            if result: return "\n".join([line[1] for line in result])
        except:
            pass
        return ""

    def _get_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def _table_to_markdown(table_data: List[List]) -> str:
        """转 Markdown + 清洗"""
        if not table_data: return ""
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
        except:
            return ""  # 结构错误兜底

        df = df.fillna("")
        # 清洗换行符
        df = df.map(lambda x: str(x).replace("\n", " ").replace("|", "\|").strip())
        return df.to_markdown(index=False, tablefmt="pipe")

    @staticmethod
    def normalize_table_header(data: List[List]) -> List[List]:
        """无头表格注入"""
        if not data: return []
        # 简单判定：第一行如果有超长文本，肯定不是Header
        if any(len(str(x)) > 20 for x in data[0]):
            new_header = [f"列{i + 1}" for i in range(len(data[0]))]
            data.insert(0, new_header)
        return data

    async def process_images(self, file_info: Dict, images: List[Dict]) -> List[Document]:
        """
                对提取的图片进行语义分析（异步并发）
                使用 VLM 模型生成图片描述
                """
        if not images:
            return []

        image_docs = []
        print(f"🖼️ 开始识别 {len(images)} 张图片内容 (使用 VLM)...")

        async def describe_single_image(img_dict):
            """内部函数：处理单张图片"""
            image_name = img_dict.get("name")
            image_path = img_dict.get("path")
            file_hash = file_info.get("file_hash")
            if not image_path or not os.path.exists(image_path):
                return None

            # 使用信号量限制并发数
            async with self.semaphore:
                # 读取图片并转换为 Base64 编码
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                # 构造 Prompt，指导模型如何描述图片
                message = HumanMessage(
                    content=[
                        {"type": "text",
                         "text": "请详细描述这张图片的内容。如果是架构图，请说明组件关系；如果是流程图，请说明步骤；如果是照片，请提炼核心信息。"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                        },
                    ]
                )

                try:
                    # 调用 VLM 模型
                    response = await self.vlm_client.ainvoke([message])
                    description = response.content
                    ocr_result = img_dict.get("description")
                    if ocr_result and ocr_result.strip():
                        # 返回 Document 对象，包含描述和元数据
                        return Document(
                            page_content=f"[OCR解析结果]:{ocr_result}\n\n[图片语义分析]: {description}",
                            metadata={
                                "file_hash": file_hash,
                                "file_name": image_name,
                                "type": "image",
                                "image_path": image_path,
                                "page": img_dict.get("page", 0),
                                "source": "pdf_extraction"
                            }
                        )
                    else:
                        return Document(
                            page_content=f"[图片语义分析]: {description}",
                            metadata={
                                "file_hash": file_hash,
                                "file_name": image_name,
                                "type": "image",
                                "image_path": image_path,
                                "page": img_dict.get("page", 0),
                                "source": "pdf_extraction"
                            }
                        )
                except Exception as e:
                    print(f"❌ 图片解析失败 ({image_path}): {e}")
                    return None

        # 创建所有任务
        tasks = [describe_single_image(img) for img in images]
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)

        # 过滤掉失败的结果（None）
        image_docs = [doc for doc in results if doc is not None]
        print(f"✅ 图片语义化完成，共生成 {len(image_docs)} 条描述。")
        return image_docs

    @staticmethod
    def to_documents(parse_result: Dict) -> List[Document]:
        """
        将解析结果转换为标准的 LangChain Document 对象列表
        方便后续存储到向量数据库或用于 RAG 检索
        """
        documents = []
        file_name = parse_result["metadata"].get("file_name", "Unknown")
        file_hash = parse_result["metadata"].get("file_hash", "")
        # 将每个文本块转换为一个 Document
        for text_block in parse_result.get("texts", []):
            documents.append(Document(
                page_content=text_block["text"],
                metadata={
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "type": text_block.get("type", "Text"),
                    "page": text_block.get("page", 0),
                    "source": "pdf_extraction"
                }
            ))

        # 将每个表格转换为一个 Document
        for table in parse_result.get("tables", []):
            documents.append(Document(
                page_content=table["content"],
                metadata={
                    "file_hash": file_hash,
                    "type": "table",
                    "page": table.get("page", 0),
                    "format": "markdown",
                    "source": "pdf_extraction"
                }
            ))

        # for image in parse_result.get("saved_images", []):
        #     documents.append(Document(
        #         page_content=image["description"],
        #         metadata={
        #             "img_name": image.get("name", ""),
        #             "img_path": image.get("path", ""),
        #             "description": image.get("description", ""),
        #             "page": image.get("page", 0),
        #             "source": "pdf_extraction",
        #             "type": "image",
        #         }
        #     ))
        return documents


# 工厂函数保持不变
def get_pdf_parser(prefer_gpu: bool = True):
    return RapidPDFParser()
