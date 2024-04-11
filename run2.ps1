$ac = Join-Path $PSScriptRoot "code\ac\ac.py"
$adam = Join-Path $PSScriptRoot "code\ac\ac adam.py"
$svrg = Join-Path $PSScriptRoot "code\ac\ac value svrg.py"
$adasvrg = Join-Path $PSScriptRoot "code\ac\ac value adasvrg.py"
$adamsvrg = Join-Path $PSScriptRoot "code\ac\ac adam value svrg.py"

$beforetime = Get-Date

# run experiments

python $adamsvrg @("cartpole", "1e-2", "20", "40", "50", "-e", "2000")
python $adamsvrg @("cartpole", "3e-2", "20", "40", "50", "-e", "2000")
python $adamsvrg @("cartpole", "1e-3", "20", "40", "50", "-e", "2000")

python $adamsvrg @("cartpole", "1e-2", "20", "60", "50", "-e", "2000")
python $adamsvrg @("cartpole", "3e-2", "20", "60", "50", "-e", "2000")
python $adamsvrg @("cartpole", "1e-3", "20", "60", "50", "-e", "2000")

python $adamsvrg @("cartpole", "1e-2", "30", "60", "50", "-e", "2000")
python $adamsvrg @("cartpole", "3e-2", "30", "60", "50", "-e", "2000")
python $adamsvrg @("cartpole", "1e-3", "30", "60", "50", "-e", "2000")

python $adamsvrg @("acrobot", "1e-3", "20", "40", "50", "-e", "2000")
python $adamsvrg @("acrobot", "3e-3", "20", "40", "50", "-e", "2000")
python $adamsvrg @("acrobot", "1e-4", "20", "40", "50", "-e", "2000")

python $adamsvrg @("acrobot", "1e-3", "20", "60", "50", "-e", "2000")
python $adamsvrg @("acrobot", "3e-3", "20", "60", "50", "-e", "2000")
python $adamsvrg @("acrobot", "1e-4", "20", "60", "50", "-e", "2000")

python $adamsvrg @("acrobot", "1e-3", "30", "60", "50", "-e", "2000")
python $adamsvrg @("acrobot", "3e-3", "30", "60", "50", "-e", "2000")
python $adamsvrg @("acrobot", "1e-4", "30", "60", "50", "-e", "2000")

$aftertime = Get-Date

$beforestring = $beforetime.ToString("MM-dd HH:mm:ss")
$afterstring = $aftertime.ToString("MM-dd HH:mm:ss")

$minutes = ($aftertime - $beforetime).TotalMinutes

Write-Host "Start: $beforestring, End: $afterstring, time spent: $minutes minutes"