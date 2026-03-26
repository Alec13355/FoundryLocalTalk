using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging;

const string DefaultFoundryBaseUrl = "http://localhost:55588/v1";
const string DefaultFoundryModel = "qwen2.5-7b";
const string DefaultMcpServerUrl = "https://learn.microsoft.com/api/mcp";
const int MaxTurns = 4;
const int MaxToolResultLength = 16000;

var foundryBaseUrl = Environment.GetEnvironmentVariable("FOUNDRY_BASE_URL") ?? DefaultFoundryBaseUrl;
var foundryModel = Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? DefaultFoundryModel;
var mcpServerUrl = Environment.GetEnvironmentVariable("MCP_SERVER_URL") ?? DefaultMcpServerUrl;

var config = new Configuration
{
    AppName = "app-name",
    LogLevel = Microsoft.AI.Foundry.Local.LogLevel.Information,
    Web = new Configuration.WebService
    {
        Urls = "http://127.0.0.1:55588"
    }
};

using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Information);
});

var logger = loggerFactory.CreateLogger<Program>();

await FoundryLocalManager.CreateAsync(config, logger);
var mgr = FoundryLocalManager.Instance;
var catalog = await mgr.GetCatalogAsync();
var model = await catalog.GetModelAsync(foundryModel);

if (model == null)
{
	Console.WriteLine($"Model '{foundryModel}' not found.");
	Console.WriteLine("Available models:");

	var availableModels = await catalog.ListModelsAsync();
	foreach (var m in availableModels)
    {
        Console.WriteLine($"  alias={m.Alias}  id={m.Id}");
    }

    throw new Exception("Model not found");
}

try
{
	await model.DownloadAsync(progress =>
	{
		Console.Write($"\rDownloading model: {progress:F2}%");
		if (progress >= 100f)
		{
			Console.WriteLine();
		}
	});

	await model.LoadAsync();
	await mgr.StartWebServiceAsync();

	Console.WriteLine($"Using Foundry endpoint: {foundryBaseUrl}");
	Console.WriteLine($"Using Foundry model: {foundryModel}");
	Console.WriteLine($"Using Microsoft Learn MCP server: {mcpServerUrl}");
	Console.WriteLine();

	var endpoint = $"{foundryBaseUrl.TrimEnd('/')}/chat/completions";
	using var foundryHttpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(3) };

	var discoveredTools = await DiscoverMcpToolsAsync(mcpServerUrl);
	if (discoveredTools.Count == 0)
	{
		Console.WriteLine("No tools discovered from the Microsoft Learn MCP server.");
		return;
	}

	var discoveredToolNames = discoveredTools
		.Select(t => t.Name)
		.ToHashSet(StringComparer.Ordinal);

	Console.WriteLine($"Discovered {discoveredTools.Count} MCP tool(s): {string.Join(", ", discoveredToolNames)}");

	var tools = discoveredTools
		.Select(t => new
		{
			type = "function",
			function = new
			{
				name = t.Name,
				description = string.IsNullOrWhiteSpace(t.Description)
					? $"Invoke MCP tool '{t.Name}'."
					: t.Description,
				parameters = JsonSerializer.Deserialize<object>(t.InputSchemaJson)
					?? new { type = "object", properties = new { } }
			}
		})
		.Cast<object>()
		.ToArray();

	var messages = new List<Dictionary<string, object?>>
	{
		new()
		{
			["role"] = "system",
			["content"] = "Use discovered MCP tools when current Microsoft documentation data is required. If a tool is needed, call it before answering and cite URLs from tool results."
		},
		new()
		{
			["role"] = "user",
			["content"] = "What is the Azure OpenAI models and regions for Foundry Agent Service? Summarize in markdown with links to relevant Microsoft documentation. Use tools if needed to get the most current information."
		}
	};

	Console.WriteLine("Starting Foundry tool-calling loop...");

	for (var turn = 1; turn <= MaxTurns; turn++)
	{
		Console.WriteLine($"Turn {turn}: sending request to Foundry...");

		var foundryRequest = new
		{
			model = foundryModel,
			stream = false,
			temperature = 0.1,
			tool_choice = "auto",
			tools,
			messages
		};

		HttpResponseMessage response;
		string responseText;
		try
		{
			response = await foundryHttpClient.PostAsJsonAsync(endpoint, foundryRequest);
			responseText = await response.Content.ReadAsStringAsync();
		}
		catch (Exception ex)
		{
			Console.WriteLine($"Foundry request failed: {ex.Message}");
			return;
		}

		if (!response.IsSuccessStatusCode)
		{
			Console.WriteLine($"Foundry request failed: {(int)response.StatusCode} {response.StatusCode}");
			Console.WriteLine(responseText);
			return;
		}

		using var json = JsonDocument.Parse(responseText);
		var message = json.RootElement
			.GetProperty("choices")[0]
			.GetProperty("message");

		if (message.TryGetProperty("tool_calls", out var toolCalls) &&
			toolCalls.ValueKind == JsonValueKind.Array &&
			toolCalls.GetArrayLength() > 0)
		{
			Console.WriteLine($"Foundry requested {toolCalls.GetArrayLength()} tool call(s).");

			var assistantMessage = new Dictionary<string, object?>
			{
				["role"] = "assistant",
				["content"] = message.TryGetProperty("content", out var assistantContent)
					? assistantContent.GetString()
					: null,
				["tool_calls"] = JsonSerializer.Deserialize<object>(toolCalls.GetRawText())
			};
			messages.Add(assistantMessage);

			foreach (var toolCall in toolCalls.EnumerateArray())
			{
				var callId = toolCall.GetProperty("id").GetString() ?? Guid.NewGuid().ToString();
				var functionName = toolCall.GetProperty("function").GetProperty("name").GetString() ?? string.Empty;
				var argsJson = toolCall.GetProperty("function").GetProperty("arguments").GetString() ?? "{}";

				Console.WriteLine($"Executing tool call {callId}: {functionName}");

				string toolResult;
				if (!discoveredToolNames.Contains(functionName))
				{
					toolResult = "{\"error\":\"Unsupported tool requested by model\"}";
				}
				else
				{
					var arguments = ParseToolArguments(argsJson);
					var mcpResult = await CallMcpToolAsync(mcpServerUrl, functionName, arguments);
					toolResult = mcpResult ?? "{\"error\":\"MCP tool call failed\"}";
				}

				if (toolResult.Length > MaxToolResultLength)
				{
					toolResult = toolResult[..MaxToolResultLength];
				}

				messages.Add(new Dictionary<string, object?>
				{
					["role"] = "tool",
					["tool_call_id"] = callId,
					["content"] = toolResult
				});
			}

			continue;
		}

		var finalAnswer = message.TryGetProperty("content", out var content) ? content.GetString() : null;
		if (string.IsNullOrWhiteSpace(finalAnswer))
		{
			Console.WriteLine("Foundry returned no final content.");
			return;
		}

		Console.WriteLine();
		Console.WriteLine("AI Foundry model availability (tool-grounded answer):");
		Console.WriteLine(finalAnswer);
		return;
	}

	Console.WriteLine("Reached max tool-calling turns without a final answer.");
}
finally
{
    await model.UnloadAsync();
    await mgr.StopWebServiceAsync();
    mgr.Dispose();
}

static object ParseToolArguments(string argumentsJson)
{
	try
	{
		return JsonSerializer.Deserialize<object>(argumentsJson) ?? new { };
	}
	catch
	{
		return new { };
	}
}

static async Task<List<McpToolDefinition>> DiscoverMcpToolsAsync(string mcpServerUrl)
{
	using var httpClient = new HttpClient();
	httpClient.Timeout = TimeSpan.FromMinutes(2);
	httpClient.DefaultRequestHeaders.Add("Accept", "application/json, text/event-stream");
	httpClient.DefaultRequestHeaders.Add("User-Agent", "FoundryLocalExample5/1.0");

	var initRequest = new
	{
		jsonrpc = "2.0",
		id = 1,
		method = "initialize",
		@params = new
		{
			protocolVersion = "2024-11-05",
			capabilities = new { },
			clientInfo = new { name = "FoundryLocalExample5", version = "1.0" }
		}
	};

	try
	{
		var initResponse = await httpClient.PostAsJsonAsync(mcpServerUrl, initRequest);
		var initText = await initResponse.Content.ReadAsStringAsync();
		if (!initResponse.IsSuccessStatusCode || ParseSseResponse(initText) == null)
		{
			return new List<McpToolDefinition>();
		}

		var listRequest = new
		{
			jsonrpc = "2.0",
			id = 2,
			method = "tools/list",
			@params = new { }
		};

		var listResponse = await httpClient.PostAsJsonAsync(mcpServerUrl, listRequest);
		var listText = await listResponse.Content.ReadAsStringAsync();
		if (!listResponse.IsSuccessStatusCode)
		{
			return new List<McpToolDefinition>();
		}

		var listResultJson = ParseSseResponse(listText);
		if (string.IsNullOrWhiteSpace(listResultJson))
		{
			return new List<McpToolDefinition>();
		}

		using var doc = JsonDocument.Parse(listResultJson);
		if (!doc.RootElement.TryGetProperty("tools", out var toolsElement) ||
			toolsElement.ValueKind != JsonValueKind.Array)
		{
			return new List<McpToolDefinition>();
		}

		var results = new List<McpToolDefinition>();
		foreach (var tool in toolsElement.EnumerateArray())
		{
			if (!tool.TryGetProperty("name", out var nameElement))
			{
				continue;
			}

			var name = nameElement.GetString();
			if (string.IsNullOrWhiteSpace(name))
			{
				continue;
			}

			var description = tool.TryGetProperty("description", out var descriptionElement)
				? descriptionElement.GetString() ?? string.Empty
				: string.Empty;

			var inputSchemaJson = tool.TryGetProperty("inputSchema", out var inputSchemaElement)
				? inputSchemaElement.GetRawText()
				: "{\"type\":\"object\",\"properties\":{}}";

			results.Add(new McpToolDefinition(name, description, inputSchemaJson));
		}

		return results;
	}
	catch
	{
		return new List<McpToolDefinition>();
	}
}

static async Task<string?> CallMcpToolAsync(string mcpServerUrl, string toolName, object toolInput)
{
	using var httpClient = new HttpClient();
	httpClient.Timeout = TimeSpan.FromMinutes(2);
	httpClient.DefaultRequestHeaders.Add("Accept", "application/json, text/event-stream");
	httpClient.DefaultRequestHeaders.Add("User-Agent", "FoundryLocalExample5/1.0");

	var initRequest = new
	{
		jsonrpc = "2.0",
		id = 1,
		method = "initialize",
		@params = new
		{
			protocolVersion = "2024-11-05",
			capabilities = new { },
			clientInfo = new { name = "FoundryLocalExample5", version = "1.0" }
		}
	};
	try
	{
		var initResponse = await httpClient.PostAsJsonAsync(mcpServerUrl, initRequest);
		var initText = await initResponse.Content.ReadAsStringAsync();
		if (!initResponse.IsSuccessStatusCode || ParseSseResponse(initText) == null)
		{
			return null;
		}

		var toolRequest = new
		{
			jsonrpc = "2.0",
			id = 3,
			method = "tools/call",
			@params = new
			{
				name = toolName,
				arguments = toolInput
			}
		};
		var toolResponse = await httpClient.PostAsJsonAsync(mcpServerUrl, toolRequest);
		var toolText = await toolResponse.Content.ReadAsStringAsync();
		if (!toolResponse.IsSuccessStatusCode)
		{
			return null;
		}

		return ParseSseResponse(toolText);
	}
	catch
	{
		return null;
	}
}

static string? ParseSseResponse(string sseContent)
{
	if (string.IsNullOrWhiteSpace(sseContent))
	{
		return null;
	}

	var lines = sseContent.Split(new[] { "\n", "\r\n" }, StringSplitOptions.RemoveEmptyEntries);
	foreach (var line in lines)
	{
		if (!line.StartsWith("data: ", StringComparison.OrdinalIgnoreCase))
		{
			continue;
		}

		var jsonStr = line["data: ".Length..].Trim();
		try
		{
			using var doc = JsonDocument.Parse(jsonStr);
			var root = doc.RootElement;
			if (root.TryGetProperty("result", out var resultElement))
			{
				return resultElement.ToString();
			}
			if (root.TryGetProperty("error", out _))
			{
				return null;
			}
			return jsonStr;
		}
		catch
		{
			// Ignore malformed event chunk.
		}
	}

	try
	{
		using var doc = JsonDocument.Parse(sseContent);
		var root = doc.RootElement;
		if (root.TryGetProperty("result", out var resultElement))
		{
			return resultElement.ToString();
		}

		return sseContent;
	}
	catch
	{
		return sseContent;
	}
}

sealed record McpToolDefinition(string Name, string Description, string InputSchemaJson);
