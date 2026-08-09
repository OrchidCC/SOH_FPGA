$ErrorActionPreference = "Stop"
$scratch = "C:\Users\orchid\AppData\Local\Temp\claude\D--CEM-REVISE\420be1a1-677e-4535-9776-d8ffa71b6c9a\scratchpad"
Add-Type -Path "$scratch\lhm\LibreHardwareMonitorLib.dll"
$comp = New-Object LibreHardwareMonitor.Hardware.Computer
$comp.IsCpuEnabled = $true
$comp.Open()

function Get-PkgPower {
    foreach ($hw in $comp.Hardware) {
        $hw.Update()
        foreach ($s in $hw.Sensors) {
            if ($s.SensorType -eq "Power" -and $s.Name -match "Package") { return [double]$s.Value }
        }
    }
    return $null
}

# sanity: list power sensors once
foreach ($hw in $comp.Hardware) { $hw.Update(); foreach ($s in $hw.Sensors) {
    if ($s.SensorType -eq "Power") { Write-Host ("SENSOR: " + $s.Name + " = " + $s.Value) } } }

# Phase 1: idle baseline, 30 s @ 2 Hz
$idle = @()
for ($i = 0; $i -lt 60; $i++) { $v = Get-PkgPower; if ($v) { $idle += $v }; Start-Sleep -Milliseconds 500 }

# Phase 2: start load, let latency phase finish (warmup+15 s), then sample 60 s @ 2 Hz
$proc = Start-Process -FilePath "python" -ArgumentList "$scratch\cpu_load.py" -PassThru -WindowStyle Hidden -RedirectStandardOutput "$scratch\cpu_lat_local.txt"
Start-Sleep -Seconds 22
$load = @()
for ($i = 0; $i -lt 120; $i++) { $v = Get-PkgPower; if ($v) { $load += $v }; Start-Sleep -Milliseconds 500 }
Stop-Process -Id $proc.Id -Force

$idleAvg = ($idle | Measure-Object -Average).Average
$loadAvg = ($load | Measure-Object -Average).Average
$result = @{
    idle_W = [math]::Round($idleAvg, 2)
    load_W = [math]::Round($loadAvg, 2)
    task_W = [math]::Round($loadAvg - $idleAvg, 2)
    idle_n = $idle.Count
    load_n = $load.Count
} | ConvertTo-Json
Write-Host $result
$result | Out-File "$scratch\cpu_power_local.json" -Encoding utf8
Get-Content "$scratch\cpu_lat_local.txt"
