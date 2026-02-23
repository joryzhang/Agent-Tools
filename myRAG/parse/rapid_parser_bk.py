"""
轻量级 PDF 解析器 - 使用 RapidOCR + PyMuPDF
适合 CPU 环境和资源受限的场景，作为 Marker+Surya 的高效替代方案
"""
# ==================== 导入必要的库 ====================
import asyncio  # 异步 I/O 支持，用于并发处理
import base64  # 用于将图片编码为 Base64 字符串供 VLM 使用
import hashlib  # 用于计算文件 MD5 哈希值，作为缓存键
import re  # 正则表达式支持
import os  # 操作系统接口，文件路径操作
import sys  # 系统相关参数
import json  # JSON 数据处理，用于缓存存取
from typing import List, Any, Dict, Optional  # 类型标注支持

import pandas as pd

# ==================== 修复 Windows 终端编码问题 ====================
# Windows 默认终端编码可能导致中文输出乱码，这里强制设置为 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==================== 导入第三方库 ====================
import fitz  # PyMuPDF 库，用于高效提取 PDF 文本、图片和表格结构
from rapidocr_onnxruntime import RapidOCR  # RapidOCR 库，基于 ONNX Runtime 的轻量级 OCR 引擎
from langchain_core.documents import Document  # LangChain 文档对象定义
from langchain_core.messages import HumanMessage  # LangChain 消息对象定义

# ==================== 设置项目路径并导入自定义模块 ====================
# 将项目根目录添加到 python path，以便导入 models 和 app 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import get_openai_virtual_model_client  # 获取 OpenAI 兼容的模型客户端
from app.config import DATA_DIR, IMG_OUTPUT_DIR  # 导入数据存储和图片输出目录配置
from dotenv import load_dotenv  # 导入环境变量加载工具

# 加载 .env 文件中的环境变量
load_dotenv()


class RapidPDFParser:
    """
    轻量级 PDF 解析器类
    主要功能：
    1. 使用 PyMuPDF 快速提取文本层
    2. 使用 PyMuPDF 查找和提取表格
    3. 提取 PDF 中的图片
    4. 使用 RapidOCR 处理扫描版页面或图片中的文字
    5. 使用 VLM (视觉语言模型) 生成图片描述
    6. 使用 LLM 生成表格摘要
    """

    def __init__(self, use_ocr: bool = True, full_ocr: bool = False):
        """
        初始化解析器

        Args:
            use_ocr: 是否启用 OCR 功能。如果为 True，会对扫描页和图片进行文字识别。
        """
        print("🔄 正在初始化 RapidOCR 解析器...")

        self.use_ocr = use_ocr
        self.full_ocr = full_ocr
        if use_ocr:
            # 初始化 RapidOCR 引擎，这会加载 ONNX 模型
            # RapidOCR 比 Tesseract 更快且对中文支持更好
            self.ocr = RapidOCR()
        else:
            self.ocr = None

        # 设置图片输出目录
        self.img_dir = str(IMG_OUTPUT_DIR)
        # 设置缓存目录，用于存储解析结果 JSON
        self.cache_dir = DATA_DIR / "cache"

        # 确保目录存在，如果不存在则创建
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        # 初始化 VLM 客户端，用于后续的图片内容理解
        self.vlm_client = get_openai_virtual_model_client()
        # 创建信号量，限制并发请求数为 5，避免 API 限流
        self.semaphore = asyncio.Semaphore(5)

        print("✅ 解析器初始化完成")

    def _get_file_hash(self, file_path: str) -> str:
        """
        计算文件的 MD5 哈希值
        用于生成缓存文件名，确同一个文件只解析一次
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 分块读取文件，每次读取 4096 字节，避免大文件占用过多内存
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _ocr_image(self, image_bytes: bytes) -> str:
        """
        对单个图片二进制数据进行 OCR 识别

        Args:
            image_bytes: 图片的二进制数据

        Returns:
            识别出的文本内容，如果识别失败则返回空字符串
        """
        if not self.ocr:
            return ""
        try:
            # 调用 RapidOCR 进行识别
            result, _ = self.ocr(image_bytes)
            if result:
                # result 是一个列表，每一项包含 [坐标, 文本, 置信度] 上面result,_是对ocr的解包，只取我们需要的result，后面的用_不要
                # 我们只需要提取文本内容并用换行符连接
                return "\n".join([line[1] for line in result])
        except Exception as e:
            print(f"  ⚠️ OCR 识别失败: {e}")
        return ""

    def _commit_table(self, pending_table: Dict, all_tables: List, page_elements: List):
        """
        [新增] 将缓冲区表格提交到结果集
        """
        if not pending_table or not pending_table["data"]:
            return

        # 1. 转 Markdown
        md_table = self._table_to_markdown(pending_table["data"])

        # 3. 存入 all_tables
        all_tables.append({
            "page": pending_table["page_start"],
            "index": len(all_tables),
            "content": md_table,
            "format": "markdown"
        })

        # 4. (可选) 将其插入到页面元素流中
        # 注意：这里传进来的 page_elements 是当前页的。
        # 如果这个表格跨页了，它应该出现在它开始的那一页。
        # 这是一个逻辑难点。为了简单，我们选择：
        # **只将表格追加到它开始的那一页的 Markdown 文本末尾**
        # 所以这里不需要操作 page_elements，而是在 parse_pdf 循环外的处理逻辑里做。

        print(f"    ✅ 提交表格 (页码 {pending_table['page_start']}, 行数 {len(pending_table['data'])})")

    def parse_pdf(self, pdf_path: str, file_name: Optional[str], ocr_threshold: int = 50) -> Dict[str, Any]:
        """
        执行 PDF 解析的主流程

        Args:
            pdf_path: PDF 文件的路径
            ocr_threshold: OCR 触发阈值。如果某页提取的文本字符数少于此值，
                           则认为该页可能是扫描图片，会尝试使用 OCR。

        Returns:
            包含所有解析结果的字典
        """
        # 计算文件哈希，用于缓存控制
        file_hash = self._get_file_hash(pdf_path)
        # 构造缓存文件路径
        cache_path = self.cache_dir / f"{file_hash}_rapid.json"

        # -------------------- 检查缓存 --------------------
        if cache_path.exists():
            print(f"♻️ 发现缓存，直接读取: {file_hash}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"🚀 使用 RapidOCR+PyMuPDF 解析: {pdf_path}")

        # 使用 PyMuPDF 打开 PDF 文档
        doc = fitz.open(pdf_path)

        # 初始化结果容器
        all_texts = []  # 存储所有文本块
        all_tables = []  # 存储所有表格
        saved_images = []  # 存储所有保存的图片信息
        markdown_parts = []  # 存储每一页转换后的 Markdown 文本
        # === [核心修改 1] 初始化表格缓冲区 ===
        # 用于存储正在处理、尚未结束的表格
        pending_table = {
            "data": None,  # List[List], 累计的行数据
            "page_start": 0,  # 表格开始的页码
            "bbox_last": None,  # 上一页表格片段的 bbox (用于判断位置连续性)
            "header": None,  # 表头
            "cols_count": 0  # 列数
        }
        # 用于跨页表格修复：记录上一页最后一个表格的表头
        last_table_header = None

        # -------------------- 逐页处理 --------------------
        for page_num in range(len(doc)):
            page = doc[page_num]  # 获取第 page_num 页对象
            print(f"  📄 处理第 {page_num + 1}/{len(doc)} 页...")
            # 这一页的元素暂存列表：[(y0, x0, "content_type", "content")]
            page_elements = []

            # ========== 1. 优先提取表格 (获取内容并记录区域) ==========
            # 这是一个关键步骤：我们先识别表格，稍后会把表格区域“抹除”，
            # 这样提取纯文本时就不会包含表格内容的乱码。
            table_bboxes = []
            try:
                # find_tables() 是 PyMuPDF 内置的检测功能
                # 可以提高对有线表格的识别率，增加 snap_tolerance 可以提高对歪斜表格的容忍度
                tables = page.find_tables(snap_tolerance=3)

                # 如果本页没有表格，说明上一页的 pending_table 肯定结束了
                if not tables:
                    self._commit_table(pending_table, all_tables, page_elements)
                    pending_table = {"data": None}  # 重置

                for i, table in enumerate(tables):
                    # extract() 提取表格内容为二维列表
                    current_data = table.extract()
                    if not current_data: continue
                    # 记录表格在页面上的边界框 (用于后续擦除)
                    table_box_to_confirm = table.bbox
                    table_bboxes.append(table.bbox)

                    # === [核心修改 2] 跨页合并逻辑 ===
                    is_merged = False

                    # 判断是否应该合并：
                    # 1. 缓冲区有数据
                    # 2. 当前是本页第一个表格 (i==0)
                    # 3. 当前表格位于页面顶部 (bbox.y0 < 100)
                    # 4. 列数一致
                    if (pending_table["data"] and i == 0 and
                            table.bbox[1] < 150 and
                            len(current_data[0]) == pending_table["cols_count"]):
                        print(f"    🔗 检测到跨页表格 (第{page_num + 1}页)，正在合并...")
                        # 特殊处理：如果接续页的第一行左侧为空 (如你的图3)，说明是上一行内容的延续
                        # 或者是重复表头，需要去掉
                        first_row = current_data[0]

                        # 情况 A: 重复表头 -> 删掉第一行，直接拼
                        if first_row == pending_table["header"]:
                            current_data = current_data[1:]
                        # 情况 B: 左侧为空 (内容延续) -> 拼接到上一行
                        elif str(first_row[0]).strip() == "":
                            # 取出 pending_table 的最后一行
                            last_row = pending_table["data"][-1]
                            # 将当前页第一行的非空内容拼接到上一行对应列
                            for col_idx, cell_val in enumerate(first_row):
                                if cell_val and str(cell_val).strip():
                                    # 拼接文本
                                    pending_table["data"][-1][col_idx] = str(last_row[col_idx]) + str(cell_val)
                            # 删掉已经被合并的第一行
                            current_data = current_data[1:]

                        # 将剩余行追加到缓冲区
                        if current_data:
                            pending_table["data"].extend(current_data)

                        # 更新状态，标记合并成功
                        pending_table["bbox_last"] = table.bbox
                        is_merged = True

                    # === [核心修改 3] 新表格处理 ===
                    if not is_merged:
                        # 如果之前有没保存的表格，先保存
                        if pending_table["data"]:
                            self._commit_table(pending_table, all_tables, page_elements)

                        # 开始新的缓冲区
                        # 1. 规范化表头 (Headless check)
                        current_data = self.normalize_table_header(current_data)

                        pending_table = {
                            "data": current_data,
                            "page_start": page_num + 1,
                            "bbox_last": table.bbox,
                            "header": current_data[0],  # 记录表头用于比对
                            "cols_count": len(current_data[0])
                        }
            except Exception as e:
                print(f"    ⚠️ 表格提取失败: {e}")

            # ========== 2. 提取图片 (在擦除表格之前) ==========
            # 必须在擦除表格之前做，防止表格背景图或其他重叠元素被误删
            try:
                image_list = page.get_image_info(xrefs=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info["xref"]  # 获取图片的交叉引用 ID

                    # 尝试处理图片（有些图片可能是很多小块拼成的，这里简单处理）
                    try:
                        base_image = doc.extract_image(xref)
                    except Exception:
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # 过滤太小的图标或线条 (小于 2KB)
                    if img_info["size"] < 2048:
                        continue

                    # 生成唯一文件名
                    img_name = f"{file_hash}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                    img_path = os.path.join(self.img_dir, img_name)
                    img_bbox = img_info["bbox"]
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)

                    current_image_data = {
                        "name": img_name,
                        "path": img_path,
                        "page": page_num + 1,
                        "description": None
                    }

                    # 如果开启 OCR，尝试识别图片中的文字
                    img_text = None
                    if self.use_ocr:
                        img_text = self._ocr_image(image_bytes)
                        if img_text.strip():
                            current_image_data["description"] = img_text  # 直接更新字典
                            all_texts.append({
                                "page": page_num + 1,
                                "type": "ImageText",
                                "text": img_text,
                                "source": img_name
                            })
                    saved_images.append(current_image_data)  # 将完整的字典加入列表
                    page_elements.append((img_bbox[1], img_bbox[0], "image", img_text))
            except Exception as e:
                print(f"    ⚠️ 图片提取失败: {e}")

            # ========== 3. 擦除表格区域 (Redaction) ==========
            # 将识别到的表格区域从页面内容中“移除”，
            # 这样后续 get_text 就不会再次提取到表格里的文字。
            if table_bboxes:
                for bbox in table_bboxes:
                    # 添加擦除注释
                    page.add_redact_annot(bbox)
                # 应用擦除 (content=False 表示不删除注释本身，images=0 表示不删除图片)
                # 实际上 apply_redactions() 默认会清除内容。
                # 注意：这会修改 page 对象的当前状态，后续对该 page 的 get_text 只能获取剩余内容。
                page.apply_redactions()

            # ========== 4. 提取剩余文本 (Standard Text Extraction) ==========
            # data-mining 模式通常能更好保留段落结构 这里取"blocks" 返回格式: [(x0, y0, x1, y1, "lines", block_no, block_type)]
            text_blocks = page.get_text("blocks")
            page_text = ""
            for block in text_blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                page_text += text
                # 过滤掉空的或者只有空白符的块
                if not text.strip(): continue

                # 这里的 text 已经是去除了表格内容的纯文本了
                page_elements.append((y0, x0, "text", text.strip()))

            # ========== 5. 自动检测是否需要 OCR (针对剩余区域) ==========
            # 如果剩余文本很少，有两种情况：
            # A. 页面本来就是空的或只有表格（不需要 OCR）
            # B. 页面是扫描件，表格检测失效（table_bboxes为空），需要 OCR
            # C. 页面除表格外确实没字了 (不需要 OCR)

            # 简单逻辑：如果没找到表格 且 字数少 -> 可能是扫描件 -> OCR
            # 如果找到了表格，说明是数字文档，字数少说明确实没字 -> 不 OCR
            if len(page_text.strip()) < ocr_threshold and self.full_ocr:
                # 只有当没有检测到表格，或者强制策略时才 OCR
                # 如果检测到了表格，通常意味着这是原生 PDF，不需要 OCR (除非是混合型)
                # 但为了保险，如果字数极少，还是检查一下。
                # 此时 page 已经被 redact 了表格，如果 OCR 这里，表格部分由白块替代，
                # 刚好避免了重复识别表格内容。

                print(f"    🔍 剩余文本较少({len(page_text.strip())}字符)，尝试 OCR 补充...")

                # 渲染页面为图片 (注意：此时表格区域已经是空白了)
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")

                ocr_text = self._ocr_image(img_bytes)
                if len(ocr_text) > len(page_text.strip()) + 10:  # 如果 OCR 多识别出了显著内容
                    print(f"    ✅ OCR 补充识别到 {len(ocr_text)} 字符")
                    page_elements = [e for e in page_elements if e[2] != "text"]
                    page_elements.append((0, 0, "ocr_text", ocr_text))  # OCR结果通常作为整页块
                    page_text = ocr_text

            # 保存文本内容
            if page_text.strip():
                all_texts.append({
                    "page": page_num + 1,
                    "type": "Text",
                    "text": page_text.strip()
                })
            page_elements.sort(key=lambda x: (x[0], x[1]))
            # 组装这一页的 Markdown
            page_md_parts = [f"## 第 {page_num + 1} 页"]
            for y0, x0, type_, content in page_elements:
                if type_ == "image":
                    # 如果图片 OCR 识别出了文字，将其包裹在引用块中，并注明来源
                    if content and content.strip():
                        page_md_parts.append(f"\n**[图片识别内容]:** {content.strip()}\n")
                elif type_ == "table":
                    # 这里是之前逻辑生成的 table，现在被 pending 逻辑取代了
                    # 只有当 table 被 commit 时，我们才把它加入。
                    # 由于逻辑复杂，我建议：page_elements 里不再放 table
                    # 而是把 all_tables 里的内容，根据 page_num 插进去。
                    pass
                    # 文本和保持原样
                    page_md_parts.append(content)

            markdown_parts.append("\n\n".join(page_md_parts))

        # 循环结束后，检查是否还有残留的表格
        if pending_table["data"]:
            self._commit_table(pending_table, all_tables, [])  # 这里传空list，因为我们下面统一处理

        # 保存总页数后关闭文档
        total_pages = len(doc)
        doc.close()

        # ========== [核心修复] 统一回填表格到 Markdown ==========
        # 此时 markdown_parts 已经包含了每一页的文本和图片
        # all_tables 包含了所有处理好的表格（含 page 字段）

        for table in all_tables:
            p_idx = table["page"] - 1  # 转换为列表索引
            if 0 <= p_idx < len(markdown_parts):
                # 策略：将表格追加到该页的末尾
                # 这样可以保证表格不会打断段落，且上下文连贯
                markdown_parts[p_idx] += f"\n\n{table['content']}\n\n"

        # 组合所有部分的 Markdown
        full_markdown = "\n\n".join(markdown_parts)

        # 构造最终结果字典
        result = {
            "markdown": full_markdown,
            "texts": all_texts,
            "tables": all_tables,
            "saved_images": saved_images,
            "metadata": {
                "file_name": file_name,
                "file_hash": file_hash,
                "total_pages": total_pages,
                "parser": "RapidOCR+PyMuPDF"
            }
        }

        # -------------------- 写入缓存 --------------------
        with open(cache_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False 确保中文正常显示，indent=2 格式化输出
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ 解析完成: 文本块[{len(all_texts)}], 表格[{len(all_tables)}], 图片[{len(saved_images)}]")
        print(result)
        return result

    @staticmethod
    def _table_to_markdown(table_data: List[List]) -> str:
        """
        将 list of lists 转换为标准 Markdown 表格，并清洗换行符等噪音。
        针对长文本法律文档优化。
        """
        if not table_data:
            return ""
        # 1. 转换为 DataFrame
        try:
            # 假设第一行是 Header
            headers = table_data[0]
            rows = table_data[1:]
            df = pd.DataFrame(rows, columns=headers)
        except Exception as e:
            print(f"⚠️ 表格结构异常，降级处理: {e}")
            # 如果列数对不上，Pandas 会报错，这里做一个兜底
            return ""

        # 2. 核心清洗：处理空值 + 暴力清除单元格内的换行符
        # 这一步是手写代码最难做到的
        df = df.fillna("")

        # 定义清洗函数：去掉换行，去掉多余空格，转义管道符
        def clean_cell(text):
            if not isinstance(text, str):
                return str(text)
            # 将换行符替换掉，保持 Markdown 表格的单行结构
            text = text.replace("\n", " ").replace("\r", "")
            # 转义管道符，防止破坏表格结构
            text = text.replace("|", "\|")
            # 去掉多余的连续空格
            return re.sub(r'\s+', ' ', text).strip()

        # 应用清洗到所有元素
        df = df.map(clean_cell)

        # 3. 输出 Markdown
        # index=False: 不显示行号
        return df.to_markdown(index=False, tablefmt="pipe")

    @staticmethod
    def normalize_table_header(table_data: List[List[Any]]) -> List[List[str]]:
        """
        智能检测并修复表格表头
        如果判定第一行是数据，则自动注入通用表头
        """
        if not table_data:
            return []

        first_row = table_data[0]
        # None 值处理为字符串
        safe_first_row = [str(cell) if cell is not None else "" for cell in first_row]

        is_header = True

        # --- 规则 1: 长度检测 ---
        # 如果第一行里，有任何一列的内容长度超过 20 个字，绝对不可能是表头
        # 表头通常是 "姓名" (2字) 或 "归属于母公司所有者的净利润" (13字)
        for cell_str in safe_first_row:
            if len(cell_str) > 20:
                is_header = False
                break

        # --- 规则 2: 标点检测 ---
        # 表头里几乎不可能出现句号(。)
        if is_header:
            for cell_str in safe_first_row:
                if "。" in cell_str or "；" in cell_str:
                    is_header = False
                    break

        # --- 注入逻辑 ---
        if not is_header:
            print(f"    🤖 检测到无头表格，正在注入默认表头...")
            # 针对你的这种两列式规章表格，["项目", "内容"] 是最通用的
            new_header = [f"列{i + 1}" for i in range(len(first_row))]

            table_data.insert(0, new_header)

        return table_data

    async def process_images(self, images: List[Dict]) -> List[Document]:
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
            image_path = img_dict.get("path")
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

    # async def process_tables(self, tables: List[Dict], llm_model=None) -> List[Document]:
    #     """
    #     对提取的表格转化为document对象
    #     """
    #     if not tables:
    #         return []
    #
    #     if llm_model is None:
    #         llm_model = self.vlm_client
    #
    #     table_docs = []
    #     print(f"📊 开始分析 {len(tables)} 个表格...")
    #
    #     async def summarize_single_table(table_dict):
    #         """内部函数：处理单个表格"""
    #         content = table_dict.get("content", "")
    #
    #         # 构造 Prompt，要求模型总结表格
    #         prompt = (
    #             "请根据以下表格的Markdown内容，生成一段简洁的文本摘要。"
    #             "摘要应包含表格的主要主题、列名含义以及关键数据点。"
    #             "不要输出标签，只输出纯文本描述。"
    #             f"\n\n表格内容:\n{content}"
    #         )
    #
    #         try:
    #             # 使用信号量限制并发
    #             async with self.semaphore:
    #                 # 调用 LLM
    #                 response = await llm_model.ainvoke([HumanMessage(content=prompt)])
    #                 summary = response.content
    #
    #             # 返回包含摘要的 Document
    #             return Document(
    #                 page_content=summary,
    #                 metadata={
    #                     "type": "table",
    #                     "page": table_dict.get("page", 0),
    #                     "markdown_content": content,
    #                     "source": "pdf_extraction"
    #                 }
    #             )
    #         except Exception as e:
    #             print(f"❌ 表格摘要生成失败: {e}")
    #             # 即使摘要生成失败，也返回原始表格内容
    #             return Document(
    #                 page_content=content,
    #                 metadata={
    #                     "type": "table",
    #                     "page": table_dict.get("page", 0),
    #                     "markdown_content": content,
    #                     "source": "pdf_extraction",
    #                     "error": str(e)
    #                 }
    #             )
    #
    #     tasks = [summarize_single_table(table) for table in tables]
    #     results = await asyncio.gather(*tasks)
    #
    #     table_docs = [doc for doc in results if doc is not None]
    #     print(f"✅ 表格分析完成，共生成 {len(table_docs)} 条摘要。")
    #     return table_docs

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


# ==================== 工厂函数 ====================
def get_pdf_parser(prefer_gpu: bool = True):
    """
    智能工厂函数：获取最合适的 PDF 解析器

    Args:
        prefer_gpu: 是否优先尝试使用 GPU 版本的解析器 (Marker)

    Returns:
        解析器实例 (AdvancedPDFParser 或 RapidPDFParser)
    """
    if prefer_gpu:
        try:
            # 尝试导入 PyTorch 并检查 CUDA 是否可用
            import torch
            if torch.cuda.is_available():
                # 如果有 GPU，导入并返回高级解析器(基于 Marker)
                from parser import AdvancedPDFParser
                return AdvancedPDFParser()
        except Exception as e:
            # 如果出错（如未安装 torch 或显存不足），打印警告并回退
            print(f"⚠️ GPU 解析器不可用: {e}")

    # 如果没有 GPU 或加载失败，回退到轻量级解析器 (RapidOCR)
    return RapidPDFParser()
