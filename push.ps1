param(
    [string]$Message = "update"
)

cd D:\study\code\UniMatch-V2

git remote set-url origin https://github.com/liuyx-1/UniMatch-V2.git

git add .

git status

git commit -m $Message

git push -u origin main