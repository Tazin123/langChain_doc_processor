## What is LangChain?

Langchain is a framework designed to help developers build applications powered by large language models (LLMs) in a structured and efficient way. It simplifies the development of LLM-based tools by providing a variety of components that one can combine to handle different tasks such as document retrieval, text processing, or building conversational agents. Langchain allows seamless integration of LLMs into workflows, such as generating text, summarizing information, answering questions, or interacting with external APIs.

## Components of LangChain

### 1. Prompts (for template management)

- Prompt Templates
- Output Parsers
- Prompt Optimization Techniques

### 2. Chains (for workflow automation)

- Chains are sequences of actions, where the output of one step can be used as the input for another.
- Can be used as building blocks for other chains

### 3. Models (for LLM interactions)

- Integration with various LLM providers (OpenAI, Anthropic, etc.)
- Both traditional LLMs and chat models
- Model input/output handling

### 4. Agents (for autonomous task execution)

- Decide which action to take based on the LLM’s response
- Can be used for tasks like querying APIs, running computations, or retrieving information from external sources.

### 5. Indexes (for document processing)

- Document splitting
- Text chunking
- Embedding generation
- Index structures
- Index persistence

## Retrieval augmented generation (RAG)

**Retrieval-Augmented Generation (RAG)** is an advanced approach that enhances the performance of language models by integrating document retrieval with text generation. It combines two core techniques:

1. **Document Retrieval**: Searching for and retrieving relevant documents or data from a knowledge base or external source, typically based on a user query or prompt.
2. **Text Generation**: Using a large language model (LLM) to generate coherent, context-aware text based on both the retrieved documents and the user's query.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/dfeeb6c0-d90a-4e98-b4fa-65943975283a/image.png)

### What is loaders?

**Loaders** are components used to extract data or documents from various sources and make them available for the retrieval system. 

Loaders deal with the specifies of accessing and converting data.

- Accessing:
    - Websites
    - Databases
    - You tube
    - arXiv etc.
- Data Types:
    - PDF
    - HTML
    - JSON
    - Word
    - Power point etc.
- Return a list of ‘Document’ Object

## Document Loading

1. Open Jupyter Notebook
2. Install LangChain

```python
! pip install langchain
```

After Successful installation output will be

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/0141e08f-70ee-487c-866a-f633be3ca713/image.png)

1. Environment Variable Setup:

```python
import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

Issue 1:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/5c65313b-3442-4e9e-9d22-82d06035d29e/image.png)

We can solve this issue in 2 ways:

First of all create an API key following these steps

- Go to https://platform.openai.com/
- Sign up for an account if you don't have one
- Once logged in, click on your profile icon
- Go to "View API Keys" or navigate to https://platform.openai.com/api-keys
- Click "Create new secret key"
- Copy the key immediately (it starts with "sk-")

Then,

1. Simply create a `.env` file in your project directory:

```python
import os
os.environ['OPENAI_API_KEY'] = ' Your API key here '
```

1. Method 2 (Manual Creation):
- Open Notepad
- Type: `OPENAI_API_KEY=your-new-key-here`
- Click File → Save As
- Navigate to `C:\Users\Tazin\Downloads\LLM`
- In the "Save as type" dropdown, select "All Files (*.*)"
- Name the file exactly `.env`
- Click Save
1. Then again run this shell:

```python
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

### 3. Loads the pdf

```python
from langchain.document_loaders import PyPDFLoader
```

Issue 1:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/07ae5f01-975a-4907-be12-3ea214d90a0c/image.png)

To solve the issue install this package by this command:

```python
pip install -U langchain-community
```

Then run this-

loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")

Issue 1:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/62bf6aec-3f5d-44fd-979a-25a609e5acc9/image.png)

To solve the issue install this package by this command:

```python
pip install pypdf
```

Then check the current directory:

```python
import os
print(os.getcwd())
```

Checks the files in current directory by this command:

```python
import os

# #Print current working directory

print("Current working directory:", os.getcwd())

# #List files in current directory

print("\nFiles in current directory:", os.listdir())
```

Output will be like this;

```
Current working directory: C:\Users\Shusmita\Documents\LLM

Files in current directory: ['.env.txt', '.ipynb_checkpoints', 'MachineLearning-Lecture01.pdf', 'Untitled.ipynb']
```

### Then loads the page:

```python
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
```

Each page is a `Document`.

A `Document` contains text (`page_content`) and `metadata`.

### Then check the page length:

```python
len(pages)
```

The output will be:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/9b0ea8a8-5ffc-4d7a-ae56-f7b1e569b707/image.png)

### Then print the first 500 characters of the content of the first page :

```python
page = pages[0]
print(page.page_content[0:500])
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/dafea0b8-523f-45a8-aa5d-a807470fdc07/image.png)

```python

```

### Then access the `metadata` attribute of a `page` object

```python
page.metadata
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/1c8f02f7-c34e-437b-b691-d03caa7d629a/image.png)

## For URL Loading:

```python
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
```

Issue 1: 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/c94ba123-8a80-42c1-b247-89f76576a9f1/image.png)

Solution:

```python
import os
from langchain.document_loaders import WebBaseLoader

# #Set the USER_AGENT environment variable

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'

# #Now use the WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
```

### Load the content from the specified URL

```python
docs = loader.load()
```

### Print the content from the URL:

```python
print(docs[0].page_content[:500])
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/834fccd2-cdb0-4712-b2e9-3688f3bd952a/image.png)

## For You tube Data loading:

Importing components from `langchain` to work with documents and YouTube audio 

```python
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
```

```python
url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
YoutubeAudioLoader([url],save_dir),
OpenAIWhisperParser()
)
docs = loader.load()
```

Issue 1:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/7df4ae62-c7ad-4442-9a38-16f90c5dd9c3/image.png)

To solve the issue:

```python

pip install yt_dlp
```

Issue 2:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/8598c2e9-c770-4726-a3a7-454e2722bbed/image.png)

To solve this issue:

```python
!pip install ffmpeg-python
```

Issue 2:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/de58e0ca-4691-49e8-9139-efe119eef842/image.png)

Solution:

```python
!pip install --upgrade langchain
!pip install --upgrade langchain-community
!pip install openai-whisper
!pip install ffmpeg-python
```

Then run the modified code:

```python
from langchain_community.document_loaders import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir = "docs/youtube/"

# #Create custom options dictionary with explicit ffmpeg path

custom_options = {
'format': 'bestaudio/best',
'postprocessors': [{
'key': 'FFmpegExtractAudio',
'preferredcodec': 'm4a',
}],
'ffmpeg_location': r'C:\ffmpeg\bin\ffmpeg.exe',  # Use raw string with full path to ffmpeg.exe
'prefer_ffmpeg': True
}

loader = GenericLoader(
YoutubeAudioLoader(
[url],
save_dir,
custom_options
),
OpenAIWhisperParser()
)

docs = loader.load()
```

![Screenshot (252).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/b0e4ee45-e9f0-4b84-b552-5e3778beb5b0/Screenshot_(252).png)

# Document Splitting:

Document splitting is a crucial preprocessing step in LangChain that divides large texts into smaller, manageable chunks for efficient processing by language models. Proper splitting ensures better context preservation and more accurate responses.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/a4ed62bd-356e-4610-a64d-4cf2bb9ad32c/image.png)

## Types of Splitters

LangChain provides several text splitter classes:

## **1.** `CharacterTextSplitter`:

- Simplest splitter that divides text based on a specified character count
- Splits on a single separator
- Useful for basic text splitting needs.

Use Case:

1. Simple text documents
2. Performance-critical applications
3. Texts with consistent structure

## 2. `RecursiveCharacterTextSplitter`

- Sophisticated version of the `CharacterTextSplitter`
- Most versatile and commonly recommended text splitter
- Tries to split based on different characters in a recursive manner.
- First, it will split on more prominent characters like paragraphs, then sentences, and finally individual characters.

Use Case:

1. Documents with clear paragraph structures
2. Mixed content types
3. When semantic coherence is crucial

### 3. MarkdownHeaderTextSplitter

- Specialized for Markdown documents, splits based on headers.
- Maintains document hierarchy
- Preserves header metadata
- Enables hierarchical navigation

 Use Case:

1. Wiki pages
2. Markdown-based content
3. Technical documentation
4. Documentation splitting

### 4. TokenTextSplitter

- Splits text based on token count rather than character count
- Handles special tokens
- Uses the model’s tokenizer to split the text by the number of tokens, ensuring compatibility with token limits of language models.

Use Case:

1. LLM-specific applications
2. When token count is critical
3. API cost optimization
4. Complex language processing

### 5. PythonCodeTextSplitter

- Optimized for splitting Python source code.
- Respects Python syntax
- Maintains code block integrity
- Preserves function and class boundaries

Use Case:

1. Python source code
2. Code documentation
3. API documentation
4. Code analysis tasks

### 6. SpacyTextSplitter

- Uses spacy for linguistics-aware splitting
- Multi-language support
- Sentence boundary detection
- Named entity preservation

Use Case:

1. Natural language processing
2. Text analysis tasks
3. Multi-language documents
4. When linguistic accuracy is crucial

### 7. NLTKTextSplitter

- Uses NLTK for natural language-aware splitting
- Uses NLTK's sentence tokenizer
- Supports multiple languages with NLTK models
- Preserves sentence boundaries

Use Case:

1. Academic text analysis
2. Linguistic research
3. Multi-language document processing
4. When sentence boundary detection is crucial

### 8. Language()

- A specialized splitter for programming language source code
- Maintains code block integrity
- Language-specific syntax awareness
- Handles multiple programming languages

Use Case:

1. Source code analysis
2. Code documentation generation
3. API documentation

### 9. SentenceTransformersTokenTextSplitter

- Uses sentence transformers for semantic-aware splitting
- Uses transformer models for tokenization
- Supports multiple languages through transformer models
- Maintains semantic coherence

Use Case:

1. Document summarization
2. Semantic analysis
3. Content clustering
4. Question-answering systems
5. When semantic coherence is crucial

## Selection Guide

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/57141e42-3c2d-4840-8537-5c25b548acf3/image.png)

## Common Parameters

- `chunk_size`: Target size for each text chunk
- `chunk_overlap`: Number of characters/tokens to overlap between chunks
- `length_function`: Function to measure chunk size
- `separators`: Characters to split on (in order of preference)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/fe19a2ad-3aa2-4cc4-ad2e-27875a43a190/image.png)

### 1. Set up OpenAI's API

```python
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

Issue 1:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/db78f9ad-4748-4afe-a8e2-754e4d2f5134/image.png)

Solution:

```python
#Set it directly in  code (not recommended for production)
import openai
openai.api_key = "your-api-key-goes-here"

# #Set it as an environment variable before running your script

import os
os.environ["OPENAI_API_KEY"] = "your-api-key-goes-here"

# #Use try/except to handle missing key gracefully

try:
openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
print("Please set the OPENAI_API_KEY environment variable")
# Handle the error appropriately
```

```python
pip install python-dotenv
```

```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))
```

### 2. Import text splitters from LangChain `text_splitter` module

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

```

### 3. Set the parameters:

```python
chunk_size =26
chunk_overlap = 4
```

### 4. Create two splitter instances

```python
r_splitter = RecursiveCharacterTextSplitter(
chunk_size=chunk_size,
chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
chunk_size=chunk_size,
chunk_overlap=chunk_overlap
)
```

### 3. Input Text:

```python
text1 = 'abcdefghijklmnopqrstuvwxyz’
r_splitter.split_text(text1)
```

Output:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/1796a89c-256c-4927-a8b8-fc32b985701d/image.png)

```python
text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg’
r_splitter.split_text(text2)
```

Output:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/7a33711d-ae95-4a48-ac04-a0771a418af5/image.png)

```python
text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z”
r_splitter.split_text(text3)
c_splitter.split_text(text3)
```

Output:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/11109409-12dc-404b-93f6-78cbb5b72b8b/image.png)

## **Recursive splitting details**

```python
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

len(some_text)
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/6215e808-3f73-447a-83f1-9553674fbfb3/image.png)

Reduce the chunk size a bit and add a period to our separators:

```python
r_splitter = RecursiveCharacterTextSplitter(
chunk_size=150,
chunk_overlap=0,
separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(some_text)

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)
```

### Extract text from PDF files page by page

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("MachineLearning-Lecture01.pdf")
pages = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)

len(docs)

len(pages)
```

Output:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/c1bdf103-1e8f-4d66-87d5-b67fb62f8abe/image.png)

## **Token splitting**

```python
from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
text1 = "foo bar bazzyfoo"
text_splitter.split_text(text1)
```

Output:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/81eeb6d1-330c-419c-8f44-49ba74cdb8a3/image.png)

### Split documents (PDF pages in my case) into smaller chunks based on tokens:

```python
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
docs[0]
pages[0].metadata
```

Output:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/3b46aa07-1420-498b-837a-a3c0b83e7134/image.png)

## **Context aware splitting**

### Use `MarkdownHeaderTextSplitter` to preserve header metadata in our chunks

```python
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)

md_header_splits[0]
Document(metadata={'Header 1': 'Title', 'Header 2': 'Chapter 1'}, page_content='Hi this is Jim  \nHi this is Joe')

md_header_splits[1]
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/3d1c0eb9-ab7c-43db-9466-f80faa147969/image.png)

# **Vectorstores and Embeddings**

In LangChain, **vectorstores** and **embeddings** are essential components that work together to enable retrieval-augmented generation (RAG), document search, and other memory-based tasks for large language models (LLMs).

## Vectorstores

Vectorstores are databases that store and index these vector embeddings, enabling. Once text is converted into vectors (embeddings), these vectors are stored in a **vectorstore**

- Efficient similarity search
- Nearest neighbor lookups
- Semantic retrieval operations

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/ee09243a-93d5-4288-85a3-653c70f3f288/image.png)

## **Embeddings**

**Embeddings** are numerical representations of text or any data in a high-dimensional space. These vector representations allow us to measure the similarity between texts by comparing the distances between their vectors. In LangChain, embeddings are used to convert text data into vectors, enabling efficient semantic search.

LangChain, embeddings convert text into vectors that can be:

- Compared for similarity
- Stored in vector databases
- Used for semantic search

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/64b7d1f8-c7c1-4764-ad34-4e1516786935/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/52c25de3-fed2-4419-abbe-7747009dc765/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9f04fa16-9d5d-49ed-8d00-46245484e5af/c2b1f4e3-ed66-49c6-8cd8-53af1fe054d6/image.png)
