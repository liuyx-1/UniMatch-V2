param(
    [string]$Message = "update"
)

cd D:\study\code\UniMatch-V2

git remote set-url origin https://github.com/liuyx-1/UniMatch-V2.git

git add .

$changes = git status --porcelain

if (-not $changes) {
    Write-Host "No changes to commit."
    exit 0
}

git commit -m $Message
git push -u origin main