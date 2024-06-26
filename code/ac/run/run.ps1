$p = Split-Path $PSScriptRoot -Parent

Write-Host $p 

$ac = Join-Path $p "ac.py"
$adam = Join-Path $p "ac adam.py"
$svrg = Join-Path $p "ac value svrg.py"
$adasvrg = Join-Path $p "sac value adasvrg.py"

$beforetime = Get-Date

# run experiments

python $ac @("cartpole", "1e-3", "50", "-e", "2000")
python $ac @("cartpole", "3e-3", "50", "-e", "2000")
python $ac @("cartpole", "1e-2", "50", "-e", "2000")

python $adam @("cartpole", "1e-3", "50", "-e", "2000")
python $adam @("cartpole", "3e-3", "50", "-e", "2000")
python $adam @("cartpole", "1e-2", "50", "-e", "2000")

# how will groupsize & update affect the performance?

# 20 groupsize, 40 update
python $svrg @("cartpole", "1e-3", "20", "40", "50", "-e", "2000")
python $svrg @("cartpole", "3e-3", "20", "40", "50", "-e", "2000")
python $svrg @("cartpole", "1e-2", "20", "40", "50", "-e", "2000")

# 20 groupsize, 60 update
python $svrg @("cartpole", "1e-3", "20", "60", "50", "-e", "2000")
python $svrg @("cartpole", "3e-3", "20", "60", "50", "-e", "2000")
python $svrg @("cartpole", "1e-2", "20", "60", "50", "-e", "2000")

# 30 groupsize, 60 update
python $svrg @("cartpole", "1e-3", "30", "60", "50", "-e", "2000")
python $svrg @("cartpole", "3e-3", "30", "60", "50", "-e", "2000")
python $svrg @("cartpole", "1e-2", "30", "60", "50", "-e", "2000")

# 20 groupsize, 40 update
python $adasvrg @("cartpole", "1e-3", "20", "40", "50", "-e", "2000")
python $adasvrg @("cartpole", "3e-3", "20", "40", "50", "-e", "2000")
python $adasvrg @("cartpole", "1e-2", "20", "40", "50", "-e", "2000")

# 20 groupsize, 60 update
python $adasvrg @("cartpole", "1e-3", "20", "60", "50", "-e", "2000")
python $adasvrg @("cartpole", "3e-3", "20", "60", "50", "-e", "2000")
python $adasvrg @("cartpole", "1e-2", "20", "60", "50", "-e", "2000")

# 30 groupsize, 60 update
python $adasvrg @("cartpole", "1e-3", "30", "60", "50", "-e", "2000")
python $adasvrg @("cartpole", "3e-3", "30", "60", "50", "-e", "2000")
python $adasvrg @("cartpole", "1e-2", "30", "60", "50", "-e", "2000")

# acrobot

python $ac @("acrobot", "1e-4", "50", "-e", "2000")
python $ac @("acrobot", "3e-4", "50", "-e", "2000")
python $ac @("acrobot", "1e-3", "50", "-e", "2000")

python $adam @("acrobot", "1e-4", "50", "-e", "2000")
python $adam @("acrobot", "3e-4", "50", "-e", "2000")
python $adam @("acrobot", "1e-3", "50", "-e", "2000")

# how will groupsize & update affect the performance?

# 20 groupsize, 40 update
python $svrg @("acrobot", "1e-4", "20", "40", "50", "-e", "2000")
python $svrg @("acrobot", "3e-4", "20", "40", "50", "-e", "2000")
python $svrg @("acrobot", "1e-3", "20", "40", "50", "-e", "2000")

# 20 groupsize, 60 update
python $svrg @("acrobot", "1e-4", "20", "60", "50", "-e", "2000")
python $svrg @("acrobot", "3e-4", "20", "60", "50", "-e", "2000")
python $svrg @("acrobot", "1e-3", "20", "60", "50", "-e", "2000")

# 30 groupsize, 60 update
python $svrg @("acrobot", "1e-4", "30", "60", "50", "-e", "2000")
python $svrg @("acrobot", "3e-4", "30", "60", "50", "-e", "2000")
python $svrg @("acrobot", "1e-3", "30", "60", "50", "-e", "2000")

# 20 groupsize, 40 update
python $adasvrg @("acrobot", "1e-4", "20", "40", "50", "-e", "2000")
python $adasvrg @("acrobot", "3e-4", "20", "40", "50", "-e", "2000")
python $adasvrg @("acrobot", "1e-3", "20", "40", "50", "-e", "2000")

# 20 groupsize, 60 update
python $adasvrg @("acrobot", "1e-4", "20", "60", "50", "-e", "2000")
python $adasvrg @("acrobot", "3e-4", "20", "60", "50", "-e", "2000")
python $adasvrg @("acrobot", "1e-3", "20", "60", "50", "-e", "2000")

# 30 groupsize, 60 update
python $adasvrg @("acrobot", "1e-4", "30", "60", "50", "-e", "2000")
python $adasvrg @("acrobot", "3e-4", "30", "60", "50", "-e", "2000")
python $adasvrg @("acrobot", "1e-3", "30", "60", "50", "-e", "2000")

$aftertime = Get-Date

$beforestring = $beforetime.ToString("MM-dd HH:mm:ss")
$afterstring = $aftertime.ToString("MM-dd HH:mm:ss")

$minutes = ($aftertime - $beforetime).TotalMinutes

Write-Host "Start: $beforestring, End: $afterstring, time spent: $minutes minutes"