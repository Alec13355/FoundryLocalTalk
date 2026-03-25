using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging;
using PDFtoImage;
using OpenAI.Chat;
using System.ClientModel;
using System.Diagnostics;

// OCR Document with Foundry Local
var config = new Configuration
{
    AppName = "FoundryLocalExample3-OCR",
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

// Initialize the singleton instance
await FoundryLocalManager.CreateAsync(config, logger);
var mgr = FoundryLocalManager.Instance;

// Get the model catalog
var catalog = await mgr.GetCatalogAsync();

// Get a text model for answering questions
var model = await catalog.GetModelAsync("phi-4");
var modelVariant = model.Variants.First(v => v.Info.Runtime?.DeviceType == DeviceType.CPU);
model.SelectVariant(modelVariant);

// Download the model (skips if already cached)
await model.DownloadAsync(progress =>
{
    Console.Write($"\rDownloading model: {progress:F2}%");
    if (progress >= 100f)
    {
        Console.WriteLine();
    }
});

// Load the model
await model.LoadAsync();

// Start the web service
await mgr.StartWebServiceAsync();

// Create OpenAI client
var client = new OpenAI.OpenAIClient(new ApiKeyCredential("notneeded"), new OpenAI.OpenAIClientOptions
{
    Endpoint = new Uri(config.Web.Urls + "/v1")
});

var chatClient = client.GetChatClient(model.Id);

string pdfPath = "/Users/aharrison/code/JanDotNetUG/FoundryLocalExample3/epsonManual.pdf";

Console.WriteLine("Extracting text from PDF with Tesseract OCR...\n");

// Convert PDF pages to images
using var pdfStream = File.OpenRead(pdfPath);
var images = PDFtoImage.Conversion.ToImages(pdfStream).ToList();

var allText = new System.Text.StringBuilder();

// Create temp directory for images
var tempDir = Path.Combine(Path.GetTempPath(), "ocr_" + Guid.NewGuid().ToString());
Directory.CreateDirectory(tempDir);

try
{
    // Process each page
    for (int i = 12; i < 15; i++)
    {
        Console.WriteLine($"Processing page {i + 1}/{images.Count}...");
        
        // Save image to temp file
        var imagePath = Path.Combine(tempDir, $"page_{i + 1}.png");
        using (var fileStream = File.Create(imagePath))
        {
            images[i].Encode(fileStream, SkiaSharp.SKEncodedImageFormat.Png, 100);
        }
        
        // Run tesseract OCR via command line
        var outputPath = Path.Combine(tempDir, $"page_{i + 1}");
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "/opt/homebrew/bin/tesseract",
                Arguments = $"\"{imagePath}\" \"{outputPath}\" -l eng",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        process.WaitForExit();
        
        // Read the OCR result
        var textFile = outputPath + ".txt";
        if (File.Exists(textFile))
        {
            var text = File.ReadAllText(textFile);
            allText.AppendLine($"\n--- Page {i + 1} ---");
            allText.AppendLine(text);
        }
        
        images[i].Dispose();
    }
}
finally
{
    // Clean up temp directory
    if (Directory.Exists(tempDir))
    {
        Directory.Delete(tempDir, true);
    }
}

Console.WriteLine($"\nExtracted {allText.Length} characters from {images.Count} pages.\n");

// Now use the extracted text to answer a question
var question = "Tell me about each controll pannel button and lights?";
Console.WriteLine($"Question: {question}\n");

var messages = new List<ChatMessage>
{
    new SystemChatMessage($"You are a helpful assistant. Answer questions based on the following manual:\n\n{allText}"),
    new UserChatMessage(question)
};

var response = await chatClient.CompleteChatAsync(messages);

Console.WriteLine("Answer:");
Console.WriteLine("=======");
Console.WriteLine(response.Value.Content[0].Text);

// Tidy up - stop web service and unload the model
await mgr.StopWebServiceAsync();
await model.UnloadAsync();
