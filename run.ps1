$ac = Join-Path $PSScriptRoot "code\ac\ac.py"
$adam = Join-Path $PSScriptRoot "code\ac\ac adam.py"
$svrg = Join-Path $PSScriptRoot "code\ac\ac value svrg.py"
$adasvrg = Join-Path $PSScriptRoot "code\ac\ac value adasvrg.py"

# run experiments

# cartpole
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