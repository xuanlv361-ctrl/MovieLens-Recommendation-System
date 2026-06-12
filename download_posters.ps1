$ApiKey = Read-Host "Enter your TMDb API Key"

$ProjectDir = "D:\movielens_project"
$InputCsv = Join-Path $ProjectDir "missing_posters_assignment.csv"

$PosterFolderName = ([char]0x7535) + ([char]0x5F71) + ([char]0x7167) + ([char]0x7247)
$PosterDir = Join-Path $ProjectDir $PosterFolderName

$LogCsv = Join-Path $ProjectDir "poster_download_result.csv"

# 先测试，建议先设成 20。确认能下载后再改成 ALL
$MaxDownload = 20

if (!(Test-Path $InputCsv)) {
    Write-Host "Missing CSV: $InputCsv" -ForegroundColor Red
    exit
}

if (!(Test-Path $PosterDir)) {
    New-Item -ItemType Directory -Path $PosterDir | Out-Null
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-FieldValue {
    param(
        $Row,
        [string[]]$Names
    )

    foreach ($Name in $Names) {
        foreach ($Prop in $Row.PSObject.Properties) {
            $CleanPropName = $Prop.Name.Trim().TrimStart([char]0xFEFF)

            if ($CleanPropName -eq $Name) {
                return $Prop.Value
            }
        }
    }

    return ""
}

function Remove-Movie-Year {
    param([string]$Title)

    if ([string]::IsNullOrWhiteSpace($Title)) {
        return ""
    }

    return ($Title -replace "\s*\(\d{4}\)\s*$", "").Trim()
}

function Get-Year-From-Text {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $m = [regex]::Match($Text, "(\d{4})")
    if ($m.Success) {
        return $m.Groups[1].Value
    }

    return ""
}

function Get-Title-From-Filename {
    param([string]$FileName)

    if ([string]::IsNullOrWhiteSpace($FileName)) {
        return ""
    }

    $name = [System.IO.Path]::GetFileNameWithoutExtension($FileName)

    # 支持中文括号和英文括号
    $name = $name -replace "（\d{4}）", ""
    $name = $name -replace "\(\d{4}\)", ""

    return $name.Trim()
}

function Get-TmdbPosterPath {
    param(
        [string]$MovieTitle,
        [string]$Year
    )

    $CleanTitle = Remove-Movie-Year $MovieTitle

    if ([string]::IsNullOrWhiteSpace($CleanTitle)) {
        return $null
    }

    $EncodedTitle = [uri]::EscapeDataString($CleanTitle)

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

$Rows = Import-Csv $InputCsv -Encoding UTF8

Write-Host "Detected columns:" -ForegroundColor Green
$Rows[0].PSObject.Properties.Name | ForEach-Object { Write-Host " - $_" }

$Results = @()
$Total = $Rows.Count
$Index = 0
$DownloadedCount = 0

foreach ($Row in $Rows) {
    $Index++

    if ($MaxDownload -ne "ALL" -and $DownloadedCount -ge $MaxDownload) {
        break
    }

    $MovieId = Get-FieldValue $Row @("movie_id", "MovieID", "电影ID")
    $OriginalTitle = Get-FieldValue $Row @("original_title", "title", "movie_title", "电影标题", "标题")
    $RequiredFileName = Get-FieldValue $Row @("required_filename", "filename", "file_name", "文件名")
    $Year = Get-FieldValue $Row @("year", "上映年份")
    $AssignedTo = Get-FieldValue $Row @("assigned_to", "负责人")

    if ([string]::IsNullOrWhiteSpace($OriginalTitle)) {
        $OriginalTitle = Get-Title-From-Filename $RequiredFileName
    }

    if ([string]::IsNullOrWhiteSpace($Year)) {
        $Year = Get-Year-From-Text $RequiredFileName
    }

    if ([string]::IsNullOrWhiteSpace($OriginalTitle)) {
        Write-Host "[$Index / $Total] Empty title, skipped." -ForegroundColor Red
        continue
    }

    if ([string]::IsNullOrWhiteSpace($RequiredFileName)) {
        $SafeName = $OriginalTitle -replace '[:/\\?*"<>|]', ''
        if (![string]::IsNullOrWhiteSpace($Year)) {
            $RequiredFileName = "$SafeName（$Year）.jpg"
        }
        else {
            $RequiredFileName = "$SafeName.jpg"
        }
    }

    $OutputPath = Join-Path $PosterDir $RequiredFileName

    Write-Host "[$Index / $Total] Searching: $OriginalTitle ($Year)" -ForegroundColor Cyan

    if (Test-Path $OutputPath) {
        Write-Host "Already exists, skipped: $RequiredFileName" -ForegroundColor DarkYellow

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            assigned_to = $AssignedTo
            status = "exists"
            note = "File already exists"
        }

        continue
    }

    $PosterPath = Get-TmdbPosterPath -MovieTitle $OriginalTitle -Year $Year

    if ($PosterPath -eq $null) {
        Write-Host "Poster not found: $OriginalTitle" -ForegroundColor Red

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            assigned_to = $AssignedTo
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

        $Results += [PSCustomObject]@{
            movie_id = $MovieId
            original_title = $OriginalTitle
            required_filename = $RequiredFileName
            assigned_to = $AssignedTo
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
            assigned_to = $AssignedTo
            status = "download_failed"
            note = $_.Exception.Message
        }
    }

    Start-Sleep -Milliseconds 350
}

$Results | Export-Csv $LogCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Finished test run." -ForegroundColor Green
Write-Host "Downloaded count: $DownloadedCount" -ForegroundColor Green
Write-Host "Log saved to: $LogCsv" -ForegroundColor Green
Write-Host "Poster folder: $PosterDir" -ForegroundColor Green
