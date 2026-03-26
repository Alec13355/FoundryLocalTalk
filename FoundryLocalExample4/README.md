# FoundryLocalExample4 - Foundry Local RAG Implementation

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using Foundry Local with Semantic Kernel, ONNX embeddings, and Qdrant vector database.

## Overview

This is a complete offline RAG solution that:
1. Embeds documents using local embedding models
2. Stores vectors in Qdrant for efficient similarity search
3. Retrieves relevant context based on user queries
4. Generates responses using Foundry Local's language models

## Prerequisites

Before running this project, you need:

### 1. Qdrant Vector Database
Install and run Qdrant locally:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Foundry Local
Install Foundry Local (0.5.100+) and ensure it's running on port 5273.

### 3. JINA Embedding Model
Download the ONNX-based embedding model files:

1. Create a `jina` directory in the project root
2. Download the following files from [HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v2-base-en):
   - [model.onnx](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/resolve/main/model.onnx)
   - [vocab.txt](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/resolve/main/vocab.txt)

Place both files in the `./jina/` directory.

### 4. .NET 8+
This project requires .NET 8 or later.

## Project Structure

```
FoundryLocalExample4/
├── Program.cs                      # Main application entry point
├── VectorStoreService.cs           # Qdrant vector database wrapper
├── DocumentIngestionService.cs     # Document chunking and embedding
├── RagQueryService.cs              # RAG query logic
├── foundry-local-architecture.md   # Sample document to ingest
└── jina/                           # ONNX model files (not in repo)
    ├── model.onnx
    └── vocab.txt
```

## Usage

### 1. Restore Dependencies
```bash
dotnet restore
```

### 2. Ingest Documents
Uncomment the ingestion code in Program.cs:
```csharp
await documentIngestionService.IngestDocumentAsync("./foundry-local-architecture.md", "doc1");
Console.WriteLine("Document ingested successfully!");
```

### 3. Run the Application
```bash
dotnet run
```

### 4. Query the RAG System
The application will query: "What's Foundry Local?" and display the answer based on the ingested document.

## Configuration

### Model Paths
Update these paths in Program.cs if your JINA model files are in a different location:
```csharp
var embeddModelPath = "./jina/model.onnx";
var embedVocab = "./jina/vocab.txt";
```

### Foundry Local Endpoint
If your Foundry Local instance runs on a different port, update the endpoint:
```csharp
builder.AddOpenAIChatCompletion(
    "qwen2.5-0.5b-instruct-generic-gpu", 
    new Uri("http://localhost:5273/v1"),  // Update port here
    apiKey: "", 
    serviceId: "qwen2.5-0.5b");
```

### Qdrant Configuration
Update the Qdrant endpoint and collection name if needed:
```csharp
var vectorStoreService = new VectorStoreService(
    "http://localhost:6334",  // Qdrant endpoint
    "",                       // API key (empty for local instance)
    "demodocs");              // Collection name
```

## Architecture Benefits

1. **Complete Offline Operation**: No external API dependencies
2. **Edge-Optimized**: Runs efficiently on local hardware
3. **Scalable Vector Search**: Qdrant provides high-performance similarity search
4. **Flexible Model Support**: ONNX Runtime supports multiple hardware providers
5. **Streaming Responses**: Real-time response generation

## Performance Considerations

- **Chunk Size**: 300 words with 60-word overlap balances context and performance
- **Vector Dimensions**: 768-dimensional embeddings from jina-embeddings-v2
- **Search Limit**: Retrieve top 5 most relevant chunks for context
- **Memory Management**: TTL-based model caching in Foundry Local

## NuGet Packages

This project uses:
- **Microsoft.SemanticKernel** (1.60.0) - Core AI orchestration framework
- **Microsoft.SemanticKernel.Connectors.Onnx** (1.60.0-alpha) - ONNX embedding support
- **Microsoft.SemanticKernel.Connectors.Qdrant** (1.60.0-preview) - Qdrant vector database connector
- **Qdrant.Client** (1.14.1) - Qdrant client library

## Troubleshooting

### Issue: "Connection refused" to Qdrant
Make sure Qdrant is running:
```bash
docker ps  # Check if Qdrant container is running
```

### Issue: "Model files not found"
Verify that model.onnx and vocab.txt are in the `./jina/` directory.

### Issue: Foundry Local connection failed
- Check that Foundry Local is running
- Verify the port in the logs (it may not be 5273)
- Update the endpoint in Program.cs accordingly

## Additional Resources

- [Foundry Local Documentation](https://github.com/microsoft/Foundry-Local)
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [JINA Embeddings Model](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
