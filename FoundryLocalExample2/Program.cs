using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging;
using System.Linq;
// Transcribing Audio
var config = new Configuration
{
    AppName = "FoundryLocalExample2",
    LogLevel = Microsoft.AI.Foundry.Local.LogLevel.Debug
};

using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Debug);
});
var logger = loggerFactory.CreateLogger<Program>();

// Initialize the singleton instance.
await FoundryLocalManager.CreateAsync(config, logger);
var mgr = FoundryLocalManager.Instance;

// Get the model catalog
var catalog = await mgr.GetCatalogAsync();

// Get a model using an alias and select the CPU model variant
var model = await catalog.GetModelAsync("whisper-tiny") ?? throw new System.Exception("Model not found");
var modelVariant = model.Variants.First(v => v.Info.Runtime?.DeviceType == DeviceType.CPU);
model.SelectVariant(modelVariant);

// Download the model (the method skips download if already cached)
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

// Get an audio client
var audioClient = await model.GetAudioClientAsync();

var ct = CancellationToken.None;

// Get a transcription with streaming outputs
string filePath = "/Users/aharrison/code/JanDotNetUG/FoundryLocalExample2/one-sheep-converted.wav";

Console.WriteLine("Starting transcription...");
var result = await audioClient.TranscribeAudioAsync(filePath, ct);
Console.WriteLine($"Transcription: {result.Text}");

// Tidy up - unload the model
await model.UnloadAsync();