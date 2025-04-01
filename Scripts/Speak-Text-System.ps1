param (
    [Parameter(Mandatory=$true)]
    [string]$Text,
    
    [Parameter(Mandatory=$false)]
    [string]$Language = "en-US",
    
    [Parameter(Mandatory=$false)]
    [string]$Voice = "",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile = ""
)

# Function to generate a temporary file path if none is provided
function Get-TempAudioFile {
    $tempFile = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "tars_speech_$([Guid]::NewGuid()).wav")
    return $tempFile
}

# If no output file is specified, create a temporary one
if ([string]::IsNullOrEmpty($OutputFile)) {
    $OutputFile = Get-TempAudioFile
}

Write-Host "Generating speech..."
Write-Host "Text: $Text"
Write-Host "Language: $Language"

try {
    # Create a SpeechSynthesizer object
    Add-Type -AssemblyName System.Speech
    $synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer

    # Set the voice based on language
    $voices = $synthesizer.GetInstalledVoices() | Where-Object { $_.Enabled -eq $true }
    
    Write-Host "Available voices:"
    foreach ($v in $voices) {
        Write-Host "  - $($v.VoiceInfo.Name) ($($v.VoiceInfo.Culture.Name))"
    }
    
    # Try to find a voice that matches the requested language
    $matchingVoice = $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq $Language } | Select-Object -First 1
    
    # If no matching voice is found, use the default voice
    if ($matchingVoice -ne $null) {
        $synthesizer.SelectVoice($matchingVoice.VoiceInfo.Name)
        Write-Host "Using voice: $($matchingVoice.VoiceInfo.Name)"
    } else {
        Write-Host "No voice found for language $Language, using default voice: $($synthesizer.Voice.Name)"
    }

    # Set the output to a file
    $synthesizer.SetOutputToWaveFile($OutputFile)
    
    # Speak the text
    $synthesizer.Speak($Text)
    
    # Close the synthesizer
    $synthesizer.Dispose()
    
    Write-Host "Speech saved to: $OutputFile"
    
    # Play the audio file
    $player = New-Object System.Media.SoundPlayer
    $player.SoundLocation = $OutputFile
    $player.Play()
    
    Write-Host "Speech playback started"
    
    # Wait a bit to allow the audio to play
    Start-Sleep -Seconds 5
    
    Write-Host "Speech generation completed successfully"
    
    # Return success
    exit 0
}
catch {
    Write-Host "Error generating speech: $_"
    exit 1
}
