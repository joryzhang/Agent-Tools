import hashlib
import uuid
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentChildProcessor:
    """
    处理父块和子块的逻辑
    它负责给每个文档生成唯一的 UUID，并将其切分为适合向量检索的小块。
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        # 初始化子文档切分器 (用于向量检索)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]  # 优先按段落切分
        )

    def process(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        核心逻辑：
        1. 给每个原始文档 (Parent) 生成唯一的 doc_id
        2. 将 Parent 切分为多个 Child
        3. 建立 Parent -> Child 的 ID 关联
        """
        parent_docs = []
        child_all_docs = []
        for doc in documents:
            # 1. 生成父文档 ID (这是连接 MySQL 和 Chroma 的关键)
            # //todo 看一下这里的doc_id 有没有必要拿掉
            """
              {
            'doc_id': 'faa4187b-685f-410a-8470-3ddac2b8c818',
            'image_path': 'F: \\myRAG\\data\\extracted_images\\3a8f637b86b3cb5fca09b62bab02707a_p1_img1.jpeg',
            'page': 1,
            'parent_id': 'faa4187b-685f-410a-8470-3ddac2b8c818',
            'source': 'pdf_extraction',
            'type': 'image'
        }
    ]
            """
            parent_id = str(uuid.uuid4())
            doc.metadata["doc_id"] = parent_id
            parent_docs.append(doc)

            # apend 这里我们自己定义我们子文档专属的id，替换vector中的ids
            file_hash = doc.metadata.get("file_hash", "unknown_hash")

            # 2. 切分出子文档 这里我们只切割文本类型的，图片和表格直接作为子文档
            if doc.metadata.get("type") in ["image", "table"]:
                # 图片和表格内容较短，直接作为子文档，ID指向父文档
                child_doc = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy()
                    # copy() 浅拷贝 引用还是之前的对象，即修改复制后的元素会修改之前的对象
                )
                child_doc.metadata["parent_id"] = parent_id
                # [直接写] 生成固定 ID (index 设为 0)
                unique_str = f"{file_hash}_{parent_id}_0"
                child_doc.id = hashlib.md5(unique_str.encode()).hexdigest()
                child_all_docs.append(child_doc)

            else:
                # 长文本切割
                sub_chunks = self.child_splitter.split_text(doc.page_content)  # 这里sub_chunks是一个str列表
                for i, chunk in enumerate(sub_chunks):
                    child_doc = Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()
                    )
                    # 关键：子文档记录父文档的 ID
                    child_doc.metadata["parent_id"] = parent_id
                    # [直接写] 生成固定 ID (index 设为 i)
                    unique_str = f"{file_hash}_{parent_id}_{i}"
                    child_doc.id = hashlib.md5(unique_str.encode()).hexdigest()
                    child_all_docs.append(child_doc)

        return parent_docs, child_all_docs
