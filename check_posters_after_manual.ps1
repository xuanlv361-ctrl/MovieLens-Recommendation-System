$ProjectDir = "D:\movielens_project"

# Build Chinese folder name: 电影照片
$PosterFolderName = ([char]0x7535) + ([char]0x5F71) + ([char]0x7167) + ([char]0x7247)
$PosterDir = Join-Path $ProjectDir $PosterFolderName

$MissingCsv = Join-Path $ProjectDir "missing_after_manual_check.csv"
$UnmatchedCsv = Join-Path $ProjectDir "unmatched_existing_posters.csv"
$DuplicateCsv = Join-Path $ProjectDir "duplicate_poster_files.csv"

$UItemFile = Get-ChildItem $ProjectDir -Recurse -Filter "u.item" -File | Select-Object -First 1

if ($null -eq $UItemFile) {
    Write-Host "u.item not found under D:\movielens_project" -ForegroundColor Red
    exit
}

if (!(Test-Path $PosterDir)) {
    Write-Host "Poster folder not found: $PosterDir" -ForegroundColor Red
    exit
}

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

    $m = [regex]::Match($Title, "[\(（](\d{4})[\)）]")
    if ($m.Success) {
        return $m.Groups[1].Value
    }

    return ""
}

function Remove-Year {
    param([string]$Title)

    if ([string]::IsNullOrWhiteSpace($Title)) {
        return ""
    }

    return ($Title -replace "\s*[\(（]\d{4}[\)）]\s*$", "").Trim()
}

function Get-RequiredFileName {
    param([string]$Title)

    $Year = Get-Year $Title
    $Name = Remove-Year $Title

    # Remove special characters, keep English letters, numbers and spaces
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

# 1. Read MovieLens u.item
$Encoding = [System.Text.Encoding]::GetEncoding("iso-8859-1")
$Lines = [System.IO.File]::ReadAllLines($UItemFile.FullName, $Encoding)

$Movies = @()

foreach ($Line in $Lines) {
    if ([string]::IsNullOrWhiteSpace($Line)) {
        continue
    }

    $Parts = $Line -split "\|"

    if ($Parts.Count -lt 2) {
        continue
    }

    $MovieId = $Parts[0]
    $Title = $Parts[1]
    $Year = Get-Year $Title
    $RequiredFileName = Get-RequiredFileName $Title
    $RequiredKey = Normalize-Key ([System.IO.Path]::GetFileNameWithoutExtension($RequiredFileName))

    $Movies += [PSCustomObject]@{
        movie_id = $MovieId
        original_title = $Title
        year = $Year
        required_filename = $RequiredFileName
        required_key = $RequiredKey
    }
}

# 2. Read existing posters
$PosterFiles = Get-ChildItem $PosterDir -File |
Where-Object { $_.Extension.ToLower() -in @(".jpg", ".jpeg", ".png", ".webp") }

$ExistingPosterRows = @()

foreach ($File in $PosterFiles) {
    $Key = Normalize-Key $File.BaseName

    if (![string]::IsNullOrWhiteSpace($Key)) {
        $ExistingPosterRows += [PSCustomObject]@{
            existing_filename = $File.Name
            existing_path = $File.FullName
            existing_key = $Key
        }
    }
}

$ExistingKeys = @{}
foreach ($Poster in $ExistingPosterRows) {
    if (!$ExistingKeys.ContainsKey($Poster.existing_key)) {
        $ExistingKeys[$Poster.existing_key] = @()
    }

    $ExistingKeys[$Poster.existing_key] += $Poster.existing_filename
}

# 3. Find missing movies
$Missing = @()

foreach ($Movie in $Movies) {
    if (!$ExistingKeys.ContainsKey($Movie.required_key)) {
        $Missing += [PSCustomObject]@{
            movie_id = $Movie.movie_id
            original_title = $Movie.original_title
            year = $Movie.year
            required_filename = $Movie.required_filename
            google_image_search = "https://www.google.com/search?tbm=isch&q=" + [uri]::EscapeDataString($Movie.original_title + " movie poster")
            tmdb_search = "https://www.themoviedb.org/search?query=" + [uri]::EscapeDataString($Movie.original_title)
        }
    }
}

# 4. Find unmatched existing poster files
$MovieKeys = @{}
foreach ($Movie in $Movies) {
    $MovieKeys[$Movie.required_key] = $Movie.original_title
}

$Unmatched = @()

foreach ($Poster in $ExistingPosterRows) {
    if (!$MovieKeys.ContainsKey($Poster.existing_key)) {
        $Unmatched += [PSCustomObject]@{
            existing_filename = $Poster.existing_filename
            existing_path = $Poster.existing_path
            match_status = "未匹配到 MovieLens 电影"
        }
    }
}

# 5. Find duplicate normalized poster keys
$Duplicates = @()

foreach ($Key in $ExistingKeys.Keys) {
    if ($ExistingKeys[$Key].Count -gt 1) {
        foreach ($FileName in $ExistingKeys[$Key]) {
            $Duplicates += [PSCustomObject]@{
                normalized_key = $Key
                filename = $FileName
                duplicate_count = $ExistingKeys[$Key].Count
            }
        }
    }
}

# 6. Export results
$Missing | Export-Csv $MissingCsv -NoTypeInformation -Encoding UTF8
$Unmatched | Export-Csv $UnmatchedCsv -NoTypeInformation -Encoding UTF8
$Duplicates | Export-Csv $DuplicateCsv -NoTypeInformation -Encoding UTF8

# 7. Print summary
Write-Host ""
Write-Host "========== Poster Check Result ==========" -ForegroundColor Green
Write-Host "u.item file: $($UItemFile.FullName)" -ForegroundColor Green
Write-Host "Poster folder: $PosterDir" -ForegroundColor Green
Write-Host "Total movies in MovieLens: $($Movies.Count)" -ForegroundColor Cyan
Write-Host "Existing poster image files: $($PosterFiles.Count)" -ForegroundColor Cyan
Write-Host "Matched movies: $($Movies.Count - $Missing.Count)" -ForegroundColor Cyan
Write-Host "Missing posters: $($Missing.Count)" -ForegroundColor Yellow
Write-Host "Unmatched existing poster files: $($Unmatched.Count)" -ForegroundColor Yellow
Write-Host "Duplicate poster keys: $($Duplicates.Count)" -ForegroundColor Yellow
Write-Host "Missing list saved to: $MissingCsv" -ForegroundColor Green
Write-Host "Unmatched list saved to: $UnmatchedCsv" -ForegroundColor Green
Write-Host "Duplicate list saved to: $DuplicateCsv" -ForegroundColor Green
Write-Host "========================================="
Write-Host ""

if ($Missing.Count -gt 0) {
    Write-Host "Opening missing poster list..." -ForegroundColor Yellow
    Invoke-Item $MissingCsv
}
else {
    Write-Host "No missing posters. All MovieLens movies have matched poster files." -ForegroundColor Green
}
