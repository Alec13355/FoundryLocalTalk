using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.Extensions.AI;

namespace FoundryLocalExample4;

public class RagQueryService
{
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingService;
    private readonly IChatCompletionService _chatService;
    private readonly VectorStoreService _vectorStoreService;

    public RagQueryService(
        IEmbeddingGenerator<string, Embedding<float>> embeddingService,
        IChatCompletionService chatService,
        VectorStoreService vectorStoreService)
    {
        _embeddingService = embeddingService;
        _chatService = chatService;
        _vectorStoreService = vectorStoreService;
    }

    public async Task<string> QueryAsync(string question)
    {
        // Generate query embedding
        var queryEmbeddingResult = await _embeddingService.GenerateAsync(question);
        var queryEmbedding = queryEmbeddingResult.Vector;
        
        // Search for relevant chunks
        var searchResults = await _vectorStoreService.SearchAsync(queryEmbedding, limit: 5);

        // Build context from results
        string str_context = "";
        foreach (var result in searchResults)
        {
            if (result.Payload.TryGetValue("text", out var text))
            {
                str_context += text.ToString();
            }
        }
        
        var prompt = $@"According to the question {question}, optimize and simplify the content. {str_context}";

        // Create chat history
        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("You are a helpful assistant that answers questions based on the provided context.");
        chatHistory.AddUserMessage(prompt);

        // Use non-streaming completion because some local OpenAI-compatible runtimes
        // reject streaming chat requests.
        var response = await _chatService.GetChatMessageContentAsync(chatHistory, cancellationToken: default);
        return string.IsNullOrWhiteSpace(response.Content)
            ? "I couldn't generate a response."
            : response.Content;
    }
}
