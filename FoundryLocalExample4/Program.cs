using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.Extensions.AI;
using FoundryLocalExample4;

// NOTE: Update these paths to point to your local JINA embedding model files
var embeddModelPath = "./jina/model-w-mean-pooling.onnx";
var embedVocab = "./jina/vocab.txt";

// Create and configure the Semantic Kernel
var builder = Kernel.CreateBuilder();

// Add BERT ONNX embedding generator (for local embeddings)
builder.AddBertOnnxEmbeddingGenerator(embeddModelPath, embedVocab);

// Add OpenAI-compatible chat completion service (connects to Foundry Local)
builder.AddOpenAIChatCompletion(
    "qwen2.5-0.5b-instruct-generic-gpu:4", 
    new Uri("http://localhost:54330/v1"), 
    apiKey: "", 
    serviceId: "qwen2.5-0.5b");

var kernel = builder.Build();

// Get services from kernel
var chatService = kernel.GetRequiredService<IChatCompletionService>(serviceKey: "qwen2.5-0.5b");
var embeddingService = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

// Create and initialize Vector Store Service
var vectorStoreService = new VectorStoreService(
    "http://localhost:6334",
    "",
    "demodocs");

await vectorStoreService.InitializeAsync();

// Create service instances
var documentIngestionService = new DocumentIngestionService(embeddingService, vectorStoreService);
var ragQueryService = new RagQueryService(embeddingService, chatService, vectorStoreService);

// Ingest a document (uncomment to ingest the sample document)
await documentIngestionService.IngestDocumentAsync("./foundry-local-architecture.md", "doc1");
Console.WriteLine("Document ingested successfully!");

// Query the RAG system
var question = "What's Foundry Local?";
Console.WriteLine($"Question: {question}");
Console.WriteLine("Generating answer...\n");

var answer = await ragQueryService.QueryAsync(question);
Console.WriteLine($"Answer: {answer}");

// Work around a native runtime shutdown crash in this local demo setup.
Environment.Exit(0);

