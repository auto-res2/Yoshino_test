import os
import sys
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.researchgraph.generator_subgraph.nodes.generator_node import generator_node
from src.researchgraph.utils.openai_client import openai_client

def test_generator_node_structure():
    """Test the structure of the generator_node function that was converted from litellm to openai"""
    print("Testing generator_node function structure...")
    
    source = inspect.getsource(generator_node)
    
    if "from openai import OpenAI" in open("/home/ubuntu/repos/Yoshino_test/src/researchgraph/generator_subgraph/nodes/generator_node.py").read():
        print("✅ generator_node imports OpenAI correctly")
        if "client = OpenAI()" in source and "client.chat.completions.create" in source:
            print("✅ generator_node uses OpenAI client correctly")
            return True
        else:
            print("❌ generator_node does not use OpenAI client correctly")
            return False
    else:
        print("❌ generator_node does not import OpenAI correctly")
        return False

def test_openai_client_structure():
    """Test the structure of the openai_client utility function"""
    print("Testing openai_client function structure...")
    
    if "from openai import OpenAI" in open("/home/ubuntu/repos/Yoshino_test/src/researchgraph/utils/openai_client.py").read():
        print("✅ openai_client imports OpenAI correctly")
        return True
    else:
        print("❌ openai_client does not import OpenAI correctly")
        return False

def test_generator_node_with_mock():
    """Test the generator_node function with mocked OpenAI client"""
    print("Testing generator_node function with mock...")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a mocked response from the OpenAI API."
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-mock-key-for-testing'}):
        with patch('openai.OpenAI', return_value=mock_client):
            try:
                base_method_text = "A method for image classification using convolutional neural networks."
                add_method_text_list = ["A method for transfer learning in vision tasks."]
                
                result = generator_node(base_method_text, add_method_text_list)
                
                if mock_client.chat.completions.create.called:
                    print("✅ OpenAI client was called correctly")
                    
                    args, kwargs = mock_client.chat.completions.create.call_args
                    if kwargs.get('model') == "o3-mini-2025-01-31":
                        print("✅ Model parameter was passed correctly")
                    else:
                        print("❌ Model parameter was not passed correctly")
                    
                    if 'messages' in kwargs and isinstance(kwargs['messages'], list):
                        print("✅ Messages parameter was passed correctly")
                    else:
                        print("❌ Messages parameter was not passed correctly")
                    
                    return True
                else:
                    print("❌ OpenAI client was not called")
                    return False
            except Exception as e:
                print(f"❌ generator_node test failed with error: {str(e)}")
                return False

if __name__ == "__main__":
    print("Running tests for OpenAI module conversion...")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is not set.")
        print("Running tests with mocks instead of actual API calls.")
    
    test_openai_client_structure()
    test_generator_node_structure()
    
    test_generator_node_with_mock()
