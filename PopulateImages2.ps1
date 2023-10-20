param (
    [int]$ImageCount,
    [int]$AreaThreshold
)

if ($ImageCount -eq 0) {$ImageCount = 50}
if ($AreaThreshold -eq 0) {$AreaThreshold = 25500}

function Get-MaxArea($ImgName, $csv) {
    $maxArea = 0
    $csv | ?{$_.image -eq $ImgName} | %{
        if ($_.x1, $_.x2, $_.y1, $_.y2 -ne 0) {
            $area = [math]::Abs(($_.x1 - $_.x2) * ($_.y1 - $_.y2))
            if ($area -gt $maxArea) {
                $maxArea = $area
            }
        }
    }
    return $maxArea
}

$csv = Import-Csv "$PSScriptRoot\annotations_drone.csv"
$MaxAreas = @{}
if (!(Test-Path "$PSScriptRoot\maxAreas.json")) {
    Write-Host "Max areas not previously found. Regathering them..."
    $UniqueImages = ($csv | select -Unique image).image
    $ImageCount = $UniqueImages.Count
    $i = 1
    $UniqueImages | %{
        $img = $_
        Write-Progress -Activity "Getting max image areas" -Status "Getting area of $img ($i/$ImageCount)" -PercentComplete (($i/$ImageCount)*100)
        $MaxAreas[$img] = Get-MaxArea $img $csv
        $i++
    }
    ConvertTo-Json $MaxAreas | Out-File "$PSScriptRoot\maxAreas.json"
} else {
    Write-Host "Max areas already present. Skipping collection."
    (Get-Content "$PSScriptRoot\maxAreas.json" | Out-String | ConvertFrom-Json).psobject.Properties | %{
        $MaxAreas[$_.Name] = [int]$_.Value
    }
}
#Above code executes only when the initial file is missing. From this point on, this code always executes (and faster).
$PossibleHealthy = ($MaxAreas.Keys | ?{$MaxAreas[$_] -lt $AreaThreshold -and $MaxAreas[$_] -ne 0})
$PossibleUnhealthy = ($MaxAreas.Keys | ?{$MaxAreas[$_] -ge $AreaThreshold})
$UnpackedPictures = (Get-ChildItem "$PSScriptRoot\images_drone\images_drone\" -Force).Name



Write-Host "`nMaking and Populating Heathy Set..."
$PickedHealthy = @()
for ($i = 0; $i -lt $ImageCount; $i++) {
    $alreadySelected = $true
    $sample = $null
    while ($alreadySelected) {
        $sample = Get-Random -InputObject $PossibleHealthy
        if (!$PickedHealthy.Contains($sample) -and $UnpackedPictures.Contains($sample)) {$alreadySelected = $false}
    }
    $PickedHealthy += $sample
}
Set-Content "$PSScriptRoot\healthySet.txt" -Value $null
$PickedHealthy | %{
    Add-Content "$PSScriptRoot\healthySet.txt" -Value $_
}



Write-Host "`nMaking and Populating Unhealthy Set..."
$PickedUnhealthy = @()
for ($i = 0; $i -lt $ImageCount; $i++) {
    $alreadySelected = $true
    $sample = $null
    while ($alreadySelected) {
        $sample = Get-Random -InputObject $PossibleUnhealthy
        if (!$PickedUnhealthy.Contains($sample) -and $UnpackedPictures.Contains($sample)) {$alreadySelected = $false}
    }
    $PickedUnhealthy += $sample
}
Set-Content "$PSScriptRoot\unhealthySet.txt" -Value $null
$PickedUnhealthy | %{
    Add-Content "$PSScriptRoot\unhealthySet.txt" -Value $_
}

Write-Host "`nDone" -ForegroundColor Green