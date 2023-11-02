param (
    [int]$ImageCount,
    [int]$AreaThreshold
)

if ($ImageCount -eq 0) {$ImageCount = 25}
if ($AreaThreshold -eq 0) {$AreaThreshold = 50500}

$csv = Import-Csv "$PSScriptRoot\annotations_drone.csv"
$TotalAreas = @{}
if (!(Test-Path "$PSScriptRoot\totalAreas.json")) {
    Write-Host "Total areas not previously found. Regathering them..."
    $CsvCount = $csv.Count
    $i = 1
    $csv | %{
        $img = $_.image
        Write-Progress -Activity "Getting total image areas" -Status "Processing csv entry $i/$CsvCount" -PercentComplete (($i/$CsvCount)*100)
        if ($_.x1, $_.x2, $_.y1, $_.y2 -ne 0) {
            $area = [math]::Abs(($_.x1 - $_.x2) * ($_.y1 - $_.y2))
            if ($TotalAreas[$img] -eq $null) {$TotalAreas[$img] = 0}
            $TotalAreas[$img] += $area
        }
        $i++
    }
    ConvertTo-Json $TotalAreas | Out-File "$PSScriptRoot\totalAreas.json"
} else {
    Write-Host "Total areas already present. Skipping collection."
    (Get-Content "$PSScriptRoot\TotalAreas.json" | Out-String | ConvertFrom-Json).psobject.Properties | %{
        $TotalAreas[$_.Name] = [int]$_.Value
    }
}
#Above code executes only when the initial file is missing. From this point on, this code always executes (and faster).
$PossibleHealthy = ($TotalAreas.Keys | ?{$TotalAreas[$_] -lt $AreaThreshold -and $TotalAreas[$_] -ne 0})
$PossibleUnhealthy = ($TotalAreas.Keys | ?{$TotalAreas[$_] -ge $AreaThreshold})
$UnpackedPictures = (Get-ChildItem "$PSScriptRoot\images_drone\images_drone\" -Force).Name
Write-Host "Possible healthy candidates: $($PossibleHealthy.Count)"
Write-Host "Possible unhealthy candidates: $($PossibleUnhealthy.Count)"



Set-Content "$PSScriptRoot\healthySet.txt" -Value $PossibleHealthy
Write-Host "`nMaking and Populating Heathy Set..."
$PickedHealthy = @()
for ($i = 0; $i -lt $ImageCount; $i++) {
    $sample = Get-Random -InputObject $PossibleHealthy
    $PossibleHealthy = $PossibleHealthy | ?{$_ -notmatch $sample}
    $PickedHealthy += $sample
}
$PickedHealthy | %{
    Copy-Item "$PSScriptRoot\images_drone\images_drone\$_" "$PSScriptRoot\TrainImages\healthy\"
}



Set-Content "$PSScriptRoot\unhealthySet.txt" -Value $PossibleUnhealthy
Write-Host "`nMaking and Populating Unhealthy Set..."
$PickedUnhealthy = @()
for ($i = 0; $i -lt $ImageCount; $i++) {
    $sample = Get-Random -InputObject $PossibleUnhealthy
    $PossibleUnhealthy = $PossibleUnhealthy | ?{$_ -notmatch $sample}
    $PickedUnhealthy += $sample
}
$PickedUnhealthy | %{
    Copy-Item "$PSScriptRoot\images_drone\images_drone\$_" "$PSScriptRoot\TrainImages\blighted\"
}

Write-Host "`nDone" -ForegroundColor Green