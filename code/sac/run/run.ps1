$p = Split-Path $PSScriptRoot -Parent

Write-Host $p 

$sac = Join-Path $p "sac.py"
$adam = Join-Path $p "sac adam.py"
$svrg = Join-Path $p "sac svrg.py"
$valuesvrg = Join-Path $p "sac value svrg.py"

$beforetime = Get-Date

# cartpole task

python $sac @("cartpole", "1e-4", "10", "50", "-e", "200")
python $sac @("cartpole", "3e-4", "10", "50", "-e", "200")
python $sac @("cartpole", "1e-3", "10", "50", "-e", "200")

# python $adam @("cartpole", "1e-4", "10", "50", "-e", "200")
# python $adam @("cartpole", "3e-4", "10", "50", "-e", "200")
# python $adam @("cartpole", "1e-3", "10", "50", "-e", "200")

python $svrg @("cartpole", "1e-4", "10", "20", "10", "-e", "200")
python $svrg @("cartpole", "3e-4", "10", "20", "10", "-e", "200")
python $svrg @("cartpole", "1e-3", "10", "20", "10", "-e", "200")

# python $valuesvrg @("cartpole", "1e-4", "10", "20", "50", "-e", "200")
# python $valuesvrg @("cartpole", "3e-4", "10", "20", "50", "-e", "200")
# python $valuesvrg @("cartpole", "1e-3", "10", "20", "50", "-e", "200")

# acrobot task

# python $sac @("acrobot", "1e-4", "10", "50", "-e", "200")
# python $sac @("acrobot", "3e-4", "10", "50", "-e", "200")
# python $sac @("acrobot", "1e-3", "10", "50", "-e", "200")

# python $adam @("acrobot", "1e-4", "10", "50", "-e", "200")
# python $adam @("acrobot", "3e-4", "10", "50", "-e", "200")
# python $adam @("acrobot", "1e-3", "10", "50", "-e", "200")

# python $svrg @("acrobot", "1e-4", "10", "20", "10", "-e", "200")
# python $svrg @("acrobot", "3e-4", "10", "20", "10", "-e", "200")
# python $svrg @("acrobot", "1e-3", "10", "20", "10", "-e", "200")

python $valuesvrg @("acrobot", "1e-4", "10", "20", "10", "-e", "200")
python $valuesvrg @("acrobot", "3e-4", "10", "20", "10", "-e", "200")
python $valuesvrg @("acrobot", "1e-3", "10", "20", "10", "-e", "200")

# python $svrg @("acrobot", "5e-4", "5", "10", "10", "-e", "200")