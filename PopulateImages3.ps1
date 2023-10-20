﻿param (
    [int]$ImageCount,
    [int]$AreaThreshold
)

if ($ImageCount -eq 0) {$ImageCount = 25}
if ($AreaThreshold -eq 0) {$AreaThreshold = 50500}

$csv = Import-Csv "$PSScriptRoot\annotations_drone.csv"
$MaxAreas = @{}
if (!(Test-Path "$PSScriptRoot\maxAreas.json")) {
    Write-Host "Max areas not previously found. Regathering them..."
    $CsvCount = $csv.Count
    $i = 1
    $csv | %{
        $img = $_.image
        Write-Progress -Activity "Getting max image areas" -Status "Processing csv entry $i/$CsvCount" -PercentComplete (($i/$CsvCount)*100)
        if ($_.x1, $_.x2, $_.y1, $_.y2 -ne 0) {
            $area = [math]::Abs(($_.x1 - $_.x2) * ($_.y1 - $_.y2))
            if ($MaxAreas[$img] -eq $null) {$MaxAreas[$img] = 0}
            if ($area -gt $MaxAreas[$img]) {$MaxAreas[$img] = $area}
        }
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
Write-Host "Possible healthy candidates: $($PossibleHealthy.Count)"
Write-Host "Possible unhealthy candidates: $($PossibleUnhealthy.Count)"



Write-Host "`nMaking and Populating Heathy Set..."
$PickedHealthy = @()
for ($i = 0; $i -lt $ImageCount; $i++) {
    $sample = Get-Random -InputObject $PossibleHealthy
    $PossibleHealthy = $PossibleHealthy | ?{$_ -notmatch $sample}
    $PickedHealthy += $sample
}
Set-Content "$PSScriptRoot\healthySet.txt" -Value $null
$PickedHealthy | %{
    Copy-Item "$PSScriptRoot\images_drone\images_drone\$_" "$PSScriptRoot\TrainImages\healthy\"
}



Write-Host "`nMaking and Populating Unhealthy Set..."
$PickedUnhealthy = @()
for ($i = 0; $i -lt $ImageCount; $i++) {
    $sample = Get-Random -InputObject $PossibleUnhealthy
    $PossibleUnhealthy = $PossibleUnhealthy | ?{$_ -notmatch $sample}
    $PickedUnhealthy += $sample
}
Set-Content "$PSScriptRoot\unhealthySet.txt" -Value $null
$PickedUnhealthy | %{
    Copy-Item "$PSScriptRoot\images_drone\images_drone\$_" "$PSScriptRoot\TrainImages\blighted\"
}

Write-Host "`nDone" -ForegroundColor Green