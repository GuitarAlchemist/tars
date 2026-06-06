$projects = Get-ChildItem -Path v1/src -Recurse -Include *.csproj, *.fsproj
$projects | ForEach-Object { $_.FullName.Substring((Get-Location).Path.Length + 8) } | Out-File -FilePath project_list.txt -Encoding utf8
