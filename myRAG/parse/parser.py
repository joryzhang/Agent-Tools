"""
高级 PDF 解析器 - 使用 Marker + Surya
注意：此解析器需要高性能 GPU 支持
"""
# ==================== 导入必要的标准库 ====================
import asyncio  # 异步并发库
import base64  # 图片 Base64 编码
import hashlib  # 计算 MD5 哈希
import re  # 正则表达式
import os  # 操作系统接口
import sys  # 系统参数
import json  # JSON 处理
from typing import List, Any, Dict  # 类型提示
from pathlib import Path  # 面向对象的文件路径

# ==================== 修复 Windows 终端编码问题 ====================
# Windows 终端默认可能不支持 emoji 显示，强制设置为 utf-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==================== 导入相关依赖库 ====================
from langchain_core.documents import Document  # LangChain 文档对象
from langchain_core.messages import HumanMessage  # LangChain 消息对象
# Marker 库：用于将 PDF 转换为高质量 Markdown
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# ==================== 导入项目自定义模块 ====================
from models import get_openai_virtual_model_client  # AI 模型客户端
from app.config import DATA_DIR, IMG_OUTPUT_DIR  # 目录配置
from dotenv import load_dotenv  # 环境变量加载

# 加载 .env 环境变量
load_dotenv()


class AdvancedPDFParser:
    """
    高级 PDF 解析器类
    主要功能：
    1. 使用 Marker 模型深度解析 PDF 布局
    2. 使用 Surya 模型进行 OCR 和文本行检测
    3. 生成高质量的 Markdown 输出
    4. 提取图片和表格
    """

    def __init__(self):
        """初始化解析器并加载 AI 模型"""
        # 预加载 Marker 需要的所有模型 (包括 Surya)
        # 注意：这会占用几 GB 的显存/内存，建议在应用启动时只初始化一次
        print("🔄 正在加载 Marker+Surya 模型...")

        # create_model_dict() 会下载并加载所需的 PyTorch 模型
        # 包括布局分析、OCR、公式识别等模型
        self.artifact_dict = create_model_dict()
        print("✅ 模型加载完成")

        # 配置目录
        self.img_dir = str(IMG_OUTPUT_DIR)
        self.cache_dir = DATA_DIR / "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        # 初始化 VLM 客户端 (用于图片理解)
        self.vlm_client = get_openai_virtual_model_client()
        # 限制并发数为 5
        self.semaphore = asyncio.Semaphore(5)

    def _get_file_hash(self, file_path: str) -> str:
        """
        计算文件的 MD5 值作为唯一标识
        用于缓存文件名，避免重复解析同一文件
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        使用 Marker + Surya 解析 PDF
        这是解析的主要入口函数
        
        Returns:
            字典包含:
            - markdown: 完整的 Markdown 文本
            - images: 图片对象字典
            - metadata: PDF 元数据
            - tables: 提取的表格列表
            - texts: 提取的文本块列表
        """
        file_hash = self._get_file_hash(pdf_path)
        cache_path = self.cache_dir / f"{file_hash}_marker.json"

        # -------------------- 检查缓存 --------------------
        if cache_path.exists():
            print(f"♻️ 发现缓存，直接读取: {file_hash}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                # images 无法序列化到 JSON，从缓存读取时为空
                cached["images"] = {}
                return cached

        print(f"🚀 使用 Marker+Surya 引擎解析: {pdf_path}")

        # -------------------- 创建 PDF 转换器 --------------------
        # 使用预加载的模型字典初始化转换器
        converter = PdfConverter(
            artifact_dict=self.artifact_dict,
            config={
                # 设置 OCR 语言，支持中文和英文
                "languages": ["Chinese", "English"],
            }
        )

        # -------------------- 执行转换 --------------------
        # 这一步会调用深度学习模型进行复杂的页面分析
        # 也是最耗时的一步
        rendered = converter(pdf_path)
        # rendered 是一个 MarkdownOutput 对象，包含 markdown, images, metadata
        full_text = rendered.markdown
        images = rendered.images if hasattr(rendered, 'images') else {}
        out_metadata = rendered.metadata if hasattr(rendered, 'metadata') else {}

        # -------------------- 保存图片到磁盘 --------------------
        saved_images = []
        if images:
            for img_name, img_data in images.items():
                img_path = os.path.join(self.img_dir, f"{file_hash}_{img_name}")
                # 根据图片对象类型保存
                if hasattr(img_data, 'save'):
                    # 如果是 PIL Image 对象
                    img_data.save(img_path)
                elif isinstance(img_data, bytes):
                    # 如果是二进制数据
                    with open(img_path, 'wb') as f:
                        f.write(img_data)

                saved_images.append({
                    "name": img_name,
                    "path": img_path
                })
                print(f"  📷 保存图片: {img_path}")

        # -------------------- 后处理：提取表格和文本 --------------------
        # Marker 输出的是纯 Markdown，我们需要从中解析出结构化数据

        # 从 Markdown 中提取表格
        tables = self._extract_tables_from_markdown(full_text)

        # 从 Markdown 中提取文本块 (分段)
        texts = self._extract_text_blocks(full_text)

        result = {
            "markdown": full_text,
            "images": images,
            "saved_images": saved_images,
            "metadata": out_metadata if isinstance(out_metadata, dict) else {},
            "tables": tables,
            "texts": texts
        }

        # -------------------- 写入缓存 --------------------
        # 准备可序列化的缓存数据 (去除 images 对象)
        cache_data = {
            "markdown": full_text,
            "saved_images": saved_images,
            "metadata": out_metadata if isinstance(out_metadata, dict) else {},
            "tables": tables,
            "texts": texts
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 解析完成: 文本块[{len(texts)}], 表格[{len(tables)}], 图片[{len(saved_images)}]")
        return result

    def _extract_tables_from_markdown(self, markdown_text: str) -> List[Dict]:
        """
        辅助函数：使用正则表达式从 Markdown 文本中提取表格
        返回表格列表，每个表格包含内容和索引
        """
        tables = []
        # 匹配 Markdown 表格的正则: 
        # 以 | 开头，第二行包含 |---| 或 |:---| 等对齐标识
        table_pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)'

        matches = re.findall(table_pattern, markdown_text)
        for i, match in enumerate(matches):
            tables.append({
                "index": i,
                "content": match.strip(),
                "format": "markdown"
            })

        return tables

    def _extract_text_blocks(self, markdown_text: str) -> List[Dict]:
        """
        辅助函数：将 Markdown 文本按段落分割为文本块
        并简单的识别段落类型（标题、列表、普通文本）
        """
        texts = []

        # 按空行分割段落
        paragraphs = markdown_text.split('\n\n')

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # 判断类型
            if para.startswith('#'):
                text_type = "Title"  # 标题
            elif para.startswith('- ') or para.startswith('* ') or re.match(r'^\d+\.', para):
                text_type = "ListItem"  # 列表项
            elif para.startswith('|'):
                # 跳过表格，因为已经在 _extract_tables_from_markdown 中处理
                continue
            elif para.startswith('!['):
                # 跳过图片引用
                continue
            else:
                text_type = "NarrativeText"  # 普通叙述文本

            texts.append({
                "index": i,
                "type": text_type,
                "text": para
            })

        return texts

    async def process_images(self, images: List[Dict]) -> List[Document]:
        """
        图片处理流水线：图片路径 -> Base64编码 -> VLM -> 文本描述
        此方法用于对提取出的图片进行深入的语义分析
        """
        if not images:
            return []

        image_docs = []
        print(f"🖼️ 开始识别 {len(images)} 张图片内容 (使用 VLM)...")

        async def describe_single_image(img_dict):
            """内部函数处理单张图片"""
            image_path = img_dict.get("path")
            if not image_path or not os.path.exists(image_path):
                return None

            async with self.semaphore:
                # 将图片转为 Base64
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                # 构造多模态消息
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

                    return Document(
                        page_content=f"[图片语义分析]: {description}",
                        metadata={
                            "type": "image",
                            "image_path": image_path,
                            "source": "pdf_extraction"
                        }
                    )
                except Exception as e:
                    print(f"❌ 图片解析失败 ({image_path}): {e}")
                    return None

        # 并发执行图片分析任务
        tasks = [describe_single_image(img) for img in images]
        results = await asyncio.gather(*tasks)

        image_docs = [doc for doc in results if doc is not None]
        print(f"✅ 图片语义化完成，共生成 {len(image_docs)} 条描述。")
        return image_docs

    async def process_tables(self, tables: List[Dict], llm_model=None) -> List[Document]:
        """
        表格处理流水线：Markdown表格 -> LLM -> 分析摘要
        此方法用于理解表格数据，生成自然语言摘要
        """
        if not tables:
            return []

        if llm_model is None:
            llm_model = self.vlm_client

        table_docs = []
        print(f"📊 开始分析 {len(tables)} 个表格...")

        async def summarize_single_table(table_dict):
            """内部函数处理单个表格"""
            content = table_dict.get("content", "")

            # 构造 Prompt 指导模型总结表格
            prompt = (
                "请根据以下表格的Markdown内容，生成一段简洁的文本摘要。"
                "摘要应包含表格的主要主题、列名含义以及关键数据点。"
                "不要输出标签，只输出纯文本描述。"
                f"\n\n表格内容:\n{content}"
            )

            try:
                # 调用 LLM
                async with self.semaphore:
                    response = await llm_model.ainvoke([HumanMessage(content=prompt)])
                    summary = response.content

                return Document(
                    page_content=summary,
                    metadata={
                        "type": "table",
                        "markdown_content": content,
                        "source": "pdf_extraction"
                    }
                )
            except Exception as e:
                print(f"❌ 表格摘要生成失败: {e}")
                # 失败时返回原始内容
                return Document(
                    page_content=content,
                    metadata={
                        "type": "table",
                        "markdown_content": content,
                        "source": "pdf_extraction",
                        "error": str(e)
                    }
                )

        # 并发执行表格摘要任务
        tasks = [summarize_single_table(table) for table in tables]
        results = await asyncio.gather(*tasks)

        table_docs = [doc for doc in results if doc is not None]
        print(f"✅ 表格分析完成，共生成 {len(table_docs)} 条摘要。")
        return table_docs

    def to_documents(self, parse_result: Dict) -> List[Document]:
        """
        将解析结果字典转换为 LangChain Document 列表
        以便于后续存入向量数据库
        """
        documents = []

        # 添加文本块
        for text_block in parse_result.get("texts", []):
            documents.append(Document(
                page_content=text_block["text"],
                metadata={
                    "type": text_block["type"],
                    "source": "pdf_extraction"
                }
            ))

        # 添加表格 (原始 Markdown 内容作为文本)
        for table in parse_result.get("tables", []):
            documents.append(Document(
                page_content=table["content"],
                metadata={
                    "type": "table",
                    "format": "markdown",
                    "source": "pdf_extraction"
                }
            ))

        return documents
