$ApiKey = Read-Host "975bab91d9e078f4aa6e1fe2dd2183c3"

$ProjectDir = "D:\movielens_project"

# Build Chinese folder name: 电影照片
$PosterFolderName = ([char]0x7535) + ([char]0x5F71) + ([char]0x7167) + ([char]0x7247)
$PosterDir = Join-Path $ProjectDir $PosterFolderName

$LogCsv = Join-Path $ProjectDir "poster_download_result.csv"

# Download all missing posters
$MaxDownload = "ALL"

# Find u.item automatically
$UItemFile = Get-ChildItem $ProjectDir -Recurse -Filter "u.item" -File | Select-Object -First 1

if ($null -eq $UItemFile) {
    Write-Host "u.item not found under D:\movielens_project" -ForegroundColor Red
    Write-Host "Please put u.item into D:\movielens_project or its subfolder." -ForegroundColor Red
    exit
}

Write-Host "Using u.item: $($UItemFile.FullName)" -ForegroundColor Green

if (!(Test-Path $PosterDir)) {
    New-Item -ItemType Directory -Path $PosterDir | Out-Null
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Normalize-Key {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $Text = $Text.ToLower()
    return ($Text -replace "[^a-z0-9]", "")
}

function Get-Year {
    param([string]$Title)

    if ([string]::IsNullOrWhiteSpace($Title)) {
        return ""
    }

    # Supports both English parentheses and Chinese full-width parentheses
    $m = [regex]::Match($Title, "[\(（](\d{4})[\)）]")

    if ($m.Success) {
        return ($m.Groups[1].Value -replace "[^0-9]", "")
    }

    return ""
}

function Remove-Year {
    param([string]$Title)

    if ([string]::IsNullOrWhiteSpace($Title)) {
        return ""
    }

    # Remove ending year like (1995) or （1995）
    return ($Title -replace "\s*[\(（]\d{4}[\)）]\s*$", "").Trim()
}

function Convert-MovieLens-Title-For-Search {
    param([string]$Title)

    $Name = Remove-Year $Title

    # Convert "Shawshank Redemption, The" -> "The Shawshank Redemption"
    # This improves TMDb search accuracy.
    $m = [regex]::Match($Name, "^(.*),\s*(The|A|An)$")

    if ($m.Success) {
        $Name = "$($m.Groups[2].Value) $($m.Groups[1].Value)"
    }

    return $Name.Trim()
}

function Get-RequiredFileName {
    param([string]$Title)

    $Year = Get-Year $Title
    $Name = Remove-Year $Title

    # Remove special characters. Keep only English letters, numbers and spaces.
    $Name = $Name -replace "[^A-Za-z0-9 ]", ""
    $Name = $Name -replace "\s+", " "
    $Name = $Name.Trim()

    $LeftParen = [char]0xFF08
    $RightParen = [char]0xFF09

    if (![string]::IsNullOrWhiteSpace($Year)) {
        return "$Name$LeftParen$Year$RightParen.jpg"
    }

    return "$Name.jpg"
}

function Get-TmdbPosterPath {
    param(
        [string]$MovieTitle,
        [string]$Year
    )

    if ([string]::IsNullOrWhiteSpace($MovieTitle)) {
        return $null
    }

    $SearchTitle = Convert-MovieLens-Title-For-Search $MovieTitle

    if ([string]::IsNullOrWhiteSpace($SearchTitle)) {
        return $null
    }

    $EncodedTitle = [uri]::EscapeDataString($SearchTitle)

    # Search with year first
    if (![string]::IsNullOrWhiteSpace($Year)) {
        $SearchUrl = "https://api.themoviedb.org/3/search/movie?api_key=$ApiKey&query=$EncodedTitle&year=$Year&include_adult=false&language=en-US"

        try {
            $Response = Invoke-RestMethod -Uri $SearchUrl -Method Get
            $Result = $Response.results | Where-Object { $_.poster_path -ne $null } | Select-Object -First 1

            if ($Result -ne $null) {
                return $Result.poster_path
            }
        }
        catch {
            Write-Host "Search with year failed: $MovieTitle" -ForegroundColor Yellow
        }
    }

    # Search without year
    $SearchUrlNoYear = "https://api.themoviedb.org/3/search/movie?api_key=$ApiKey&query=$EncodedTitle&include_adult=false&language=en-US"

    try {
        $Response = Invoke-RestMethod -Uri $SearchUrlNoYear -Method Get
        $Result = $Response.results | Where-Object { $_.poster_path -ne $null } | Select-Object -First 1

        if ($Result -ne $null) {
            return $Result.poster_path
        }
    }
    catch {
        Write-Host "Search without year failed: $MovieTitle" -ForegroundColor Yellow
    }

    return $null
}

# Fix existing filenames like Movie（（1995））.jpg -> Movie（1995）.jpg
function Fix-Double-Year-Parentheses {
    param([string]$FolderPath)

    $LeftParen = [char]0xFF08
    $RightParen = [char]0xFF09

    Get-ChildItem $FolderPath -File -ErrorAction SilentlyContinue | ForEach-Object {
        $OldName = $_.Name

        $NewName = [regex]::Replace(
            $OldName,
            [regex]::Escape("$LeftParen$LeftParen") + "(\d{4})" + [regex]::Escape("$RightParen$RightParen"),
            "$LeftParen`$1$RightParen"
        )

        if ($NewName -ne $OldName) {
            $NewPath = Join-Path $FolderPath $NewName

            if (!(Test-Path $NewPath)) {
                Rename-Item -LiteralPath $_.FullName -NewName $NewName
                Write-Host "Fixed filename: $OldName -> $NewName" -ForegroundColor Green
            }
        }
    }
}

Fix-Double-Year-Parentheses -FolderPath $PosterDir

# Build existing poster index
$ExistingKeys = @{}

Get-ChildItem $PosterDir -File -ErrorAction SilentlyContinue |
Where-Object { $_.Extension.ToLower() -in @(".jpg", ".jpeg", ".png", ".webp") } |
ForEach-Object {
    $key = Normalize-Key $_.BaseName

    if (![string]::IsNullOrWhiteSpace($key)) {
        $ExistingKeys[$key] = $_.FullName
    }
}

Write-Host "Existing poster files: $($ExistingKeys.Count)" -ForegroundColor Green

# Read u.item using ISO-8859-1
$Encoding = [System.Text.Encoding]::GetEncoding("iso-8859-1")
$Lines = [System.IO.File]::ReadAllLines($UItemFile.FullName, $Encoding)

$Results = @()
$Total = $Lines.Count
$Index = 0
$DownloadedCount = 0

foreach ($Line in $Lines) {
    $Index++

    if ($MaxDownload -ne "ALL" -and $DownloadedCount -ge $MaxDownload) {
        break
    }

    if ([string]::IsNullOrWhiteSpace($Line)) {
        continue
    }

    $Parts = $Line -split "\|"

    if ($Parts.Count -lt 2) {
        continue
    }

    $MovieId = $Parts[0]
    $OriginalTitle = $Parts[1]
    $Year = Get-Year $OriginalTitle
    $RequiredFileName = Get-RequiredFileName $OriginalTitle

    if ([string]::IsNullOrWhiteSpace($OriginalTitle)) {
        Write-Host "[$Index / $Total] Empty title, skipped." -ForegroundColor Red
        continue
    }

    $OutputPath = Join-Path $PosterDir $RequiredFileName
    $RequiredKey = Normalize-Key ([System.IO.Path]::GetFileNameWithoutExtension($RequiredFileName))

    if ($ExistingKeys.ContainsKey($RequiredKey) -or (Test-Path $OutputPath)) {
        Write-Host "[$Index / $Total] Exists: $OriginalTitle" -ForegroundColor DarkYellow

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            year = $Year
            status = "exists"
            note = "File already exists"
        }

        continue
    }

    Write-Host "[$Index / $Total] Searching: $OriginalTitle" -ForegroundColor Cyan

    $PosterPath = Get-TmdbPosterPath -MovieTitle $OriginalTitle -Year $Year

    if ($PosterPath -eq $null) {
        Write-Host "Poster not found: $OriginalTitle" -ForegroundColor Red

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            year = $Year
            status = "not_found"
            note = "No poster found from TMDb"
        }

        Start-Sleep -Milliseconds 350
        continue
    }

    $PosterUrl = "https://image.tmdb.org/t/p/w500$PosterPath"

    try {
        Invoke-WebRequest -Uri $PosterUrl -OutFile $OutputPath

        Write-Host "Downloaded: $RequiredFileName" -ForegroundColor Green
        $DownloadedCount++

        # Add new file to existing index immediately
        $ExistingKeys[$RequiredKey] = $OutputPath

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            year = $Year
            status = "downloaded"
            note = $PosterUrl
        }
    }
    catch {
        Write-Host "Download failed: $OriginalTitle" -ForegroundColor Red

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            year = $Year
            status = "download_failed"
            note = $_.Exception.Message
        }
    }

    Start-Sleep -Milliseconds 350
}

$Results | Export-Csv $LogCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Finished." -ForegroundColor Green
Write-Host "Downloaded count: $DownloadedCount" -ForegroundColor Green
Write-Host "Log saved to: $LogCsv" -ForegroundColor Green
Write-Host "Poster folder: $PosterDir" -ForegroundColor Green