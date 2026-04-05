param (
    [string]$RootFolder = "."
)

$folders = Get-ChildItem -Path $RootFolder -Recurse -Directory -Filter "__pycache__" |
    Where-Object { $_.FullName -notlike "*\.venv\*" }

if ($folders.Count -eq 0) {
    Write-Host "No __pycache__ folders found under '$RootFolder'." -ForegroundColor Yellow
    exit
}

Write-Host "Found $($folders.Count) __pycache__ folder(s) to delete:`n" -ForegroundColor Cyan

foreach ($folder in $folders) {
    Write-Host "  Deleting: $($folder.FullName)" -ForegroundColor Gray
    Remove-Item -Path $folder.FullName -Recurse -Force
}

Write-Host "`nDone. $($folders.Count) folder(s) deleted." -ForegroundColor Green