<#
.SYNOPSIS
  Create a new CUDA example project from templates/ProjectTemplate and add it to CudaPractice.slnx.

.EXAMPLE
  .\tools\new-project.ps1 -Stage S2_Memory -Name 04_Transpose
  -> S2_Memory\04_Transpose\04_Transpose.vcxproj, main.cu, README.md
  -> registered under solution folder /S2_Memory/

  Close the solution in Visual Studio before running (or reload when VS asks).
#>
param(
    [Parameter(Mandatory = $true)][string]$Stage,   # e.g. S2_Memory
    [Parameter(Mandatory = $true)][string]$Name     # e.g. 04_Transpose
)

$ErrorActionPreference = 'Stop'
$root     = Split-Path -Parent $PSScriptRoot
$template = Join-Path $root 'templates\ProjectTemplate'
$destDir  = Join-Path $root (Join-Path $Stage $Name)
$slnPath  = Join-Path $root 'CudaPractice.slnx'

if ($Name -notmatch '^[A-Za-z0-9_]+$')  { throw "Name must be letters/digits/underscore: $Name" }
if ($Stage -notmatch '^[A-Za-z0-9_]+$') { throw "Stage must be letters/digits/underscore: $Stage" }
if (Test-Path $destDir) { throw "Already exists: $destDir" }

$guid      = [guid]::NewGuid().ToString().ToUpper()
$namespace = 'P' + ($Name -replace '_', '')
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

New-Item -ItemType Directory -Path $destDir | Out-Null

Get-ChildItem -Path $template -File | ForEach-Object {
    $text = [System.IO.File]::ReadAllText($_.FullName, $utf8NoBom)
    $text = $text.Replace('__PROJECT_NAME__', $Name).Replace('__PROJECT_GUID__', $guid).Replace('__ROOT_NAMESPACE__', $namespace)
    $outName = $_.Name.Replace('__PROJECT_NAME__', $Name)
    [System.IO.File]::WriteAllText((Join-Path $destDir $outName), $text, $utf8NoBom)
}

# ---- register in .slnx
[xml]$sln = [System.IO.File]::ReadAllText($slnPath, $utf8NoBom)
$folderName = "/$Stage/"
$folder = $sln.Solution.Folder | Where-Object { $_.Name -eq $folderName } | Select-Object -First 1
if (-not $folder) {
    $folder = $sln.CreateElement('Folder')
    $folder.SetAttribute('Name', $folderName)
    [void]$sln.Solution.AppendChild($folder)
}
$proj = $sln.CreateElement('Project')
$proj.SetAttribute('Path', "$Stage/$Name/$Name.vcxproj")
[void]$folder.AppendChild($proj)

$settings = New-Object System.Xml.XmlWriterSettings
$settings.Indent = $true
$settings.IndentChars = '  '
$settings.OmitXmlDeclaration = $true
$settings.Encoding = $utf8NoBom
$writer = [System.Xml.XmlWriter]::Create($slnPath, $settings)
$sln.Save($writer)
$writer.Close()

Write-Host "Created  $destDir"
Write-Host "Added to CudaPractice.slnx under $folderName"
