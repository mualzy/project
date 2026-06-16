param(
    [Parameter(Mandatory=$true)]
    [string]$Name,
    [Parameter(Mandatory=$true)]
    [string]$Command,
    [string]$LiveLog = "logs/repro_20260528_live.log",
    [string]$RunLog = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath ".").Path
$livePath = Join-Path $root $LiveLog
if (-not $RunLog) {
    $RunLog = "logs/$Name.log"
}
$runPath = Join-Path $root $RunLog
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $livePath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runPath) | Out-Null

function Add-Utf8Line {
    param(
        [string]$Path,
        [string]$Line,
        [switch]$BestEffort
    )
    try {
        Add-Content -Path $Path -Encoding utf8 -Value $Line -ErrorAction Stop
    } catch {
        if (-not $BestEffort) {
            throw
        }
    }
}

$started = Get-Date
$startLine = "===== $Name START $($started.ToString('yyyy-MM-dd HH:mm:ss')) ====="
Add-Utf8Line -Path $livePath -Line $startLine -BestEffort
Add-Utf8Line -Path $livePath -Line "COMMAND: $Command" -BestEffort
Add-Utf8Line -Path $runPath -Line $startLine
Add-Utf8Line -Path $runPath -Line "COMMAND: $Command"

$ErrorActionPreference = "Continue"
& cmd.exe /c "$Command 2>&1" |
    ForEach-Object {
        $line = $_
        Write-Output $line
        Add-Utf8Line -Path $runPath -Line $line
        Add-Utf8Line -Path $livePath -Line $line -BestEffort
    }
$exitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"

$ended = Get-Date
$seconds = [Math]::Round(($ended - $started).TotalSeconds, 2)
$status = if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED exit=$exitCode" }
$endLine = "===== $Name END $($ended.ToString('yyyy-MM-dd HH:mm:ss')) runtime=${seconds}s $status ====="
Add-Utf8Line -Path $runPath -Line $endLine
Add-Utf8Line -Path $livePath -Line $endLine -BestEffort
exit $exitCode
