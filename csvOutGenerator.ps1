param (
    [Parameter(Mandatory=$true)]
    [string]$ImagesDir,
    [Parameter(Mandatory=$true)]
    [string]$TrainCSVPath
)

$csvTraining = Import-Csv $TrainCSVPath
$images = Get-ChildItem $ImagesDir -File

$imagesSampled = @()
while ($imagesSampled.Count -lt 50) {
    $entry = ($csvTraining | Get-Random).fileName
    if ($imagesSampled -notcontains $entry) {
        Copy-Item "$((Get-Item $ImagesDir).FullName)/$entry" "$PSScriptRoot/app/static/images/"
        $imagesSampled += $entry
    }
}

$csvTraining | ?{$imagesSampled -contains $_.fileName} | Export-Csv "$PSScriptRoot/app/csvOut.csv" -Force -NoTypeInformation