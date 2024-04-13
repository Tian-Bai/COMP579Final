$ac = Join-Path $PSScriptRoot "code\ac\ac.py"
$adam = Join-Path $PSScriptRoot "code\ac\ac adam.py"
$svrg = Join-Path $PSScriptRoot "code\ac\ac value svrg.py"
$adasvrg = Join-Path $PSScriptRoot "code\ac\ac value adasvrg.py"

$beforetime = Get-Date

python $svrg @("cartpole", "1e-3", "20", "100", "50", "-e", "2000")
python $svrg @("cartpole", "3e-3", "20", "100", "50", "-e", "2000")
python $svrg @("cartpole", "1e-2", "20", "100", "50", "-e", "2000")

python $adasvrg @("cartpole", "1e-3", "20", "100", "50", "-e", "2000")
python $adasvrg @("cartpole", "3e-3", "20", "100", "50", "-e", "2000")
python $adasvrg @("cartpole", "1e-2", "20", "100", "50", "-e", "2000")