import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["OPENAI_API_KEY"] = "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"

from src.researchgraph.retrieve_paper_subgraph.nodes.extract_paper_title_node import extract_paper_title_node
from src.researchgraph.retrieve_paper_subgraph.nodes.select_best_paper_node import select_best_paper_node

def test_extract_paper_title():
    """extract_paper_title_nodeの動作確認"""
    print("extract_paper_title_nodeのテスト実行中...")
    
    llm_name = "gpt-3.5-turbo"
    queries = ["deep learning"]
    scraped_results = [
        "# ICLR 2024 - Deep Learning Advances\n\nThis paper discusses recent advances in deep learning architectures and training techniques...",
        "# ICLR 2024 - Neural Networks for Vision\n\nIn this study, novel convolutional neural network designs are introduced to improve image recognition..."
    ]
    
    try:
        result = extract_paper_title_node(
            llm_name=llm_name,
            queries=queries,
            scraped_results=scraped_results
        )
        print(f"Success! Result: {result}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_select_best_paper():
    """select_best_paper_nodeの動作確認"""
    print("select_best_paper_nodeのテスト実行中...")
    
    llm_name = "gpt-3.5-turbo"
    prompt_template = """
    Extract the arxiv ID from the following content and return as JSON with the key 'selected_arxiv_id':
    {{candidate_papers}}
    """
    candidate_papers = [
        {"arxiv_id": "2201.01234", "title": "Deep Learning Paper 1"},
        {"arxiv_id": "2202.05678", "title": "Deep Learning Paper 2"}
    ]
    
    try:
        result = select_best_paper_node(
            llm_name=llm_name,
            prompt_template=prompt_template,
            candidate_papers=candidate_papers
        )
        print(f"Success! Result: {result}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("修正後のコードテスト開始...")
    
    test_results = []
    test_results.append(("extract_paper_title", test_extract_paper_title()))
    test_results.append(("select_best_paper", test_select_best_paper()))
    
    print("\nテスト結果まとめ:")
    for name, result in test_results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{name}: {status}")
