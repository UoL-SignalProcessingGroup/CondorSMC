echo --Initialising CondorSMC daemon at %TIME% %DATE%
@echo off

@REM Set up environment
set USERPROFILE=%CD%

for /F "usebackq delims=" %%a in ("session_args.txt") do (
    call :process_argument "%%a"
)

goto :continue

:process_argument
set arg=%~1
if not defined arg goto :eof

@REM Extract argument name and value
for /F "tokens=1,* delims== " %%b in ("%arg%") do (
    set "name=%%b"
    set "value=%%c"
)

@REM Remove surrounding quotes from the value
set value=%value:"=%

@REM Set the variable with the extracted value
set %name%=%value%

goto :eof

:continue

echo [PROCESS] Initialising config fil (%TIME% %DATE%)
mkdir config

if "%role%"=="follower" (
    rename follower_config.cfg config.cfg
) else if "%role%"=="manager" (
    rename manager_config.cfg config.cfg
) else (
    echo "Invalid role: %role%"
    exit /b 1
)

move config.cfg config\config.cfg

@REM Extract Conda distribution and configure path
echo [PROCESS] Activate Conda environment and upgrade pip (%TIME% %DATE%)
mkdir stan_env
tar -xzf test_env.tar.gz -C stan_env
cd ./stan_env/Scripts
call activate
cd ../../

python -m pip install -q --upgrade pip

echo [PROCESS] Installing BridgeStan (%TIME% %DATE%)
set CMDSTAN=%CD%\stan_env\Library\bin\cmdstan
set CMDSTAN=%CMDSTAN:\=/%
set BRIDGESTAN_VERSION=2.3.0
python -m pip install --upgrade bridgestan==%BRIDGESTAN_VERSION%
mkdir ".bridgestan\bridgestan-%BRIDGESTAN_VERSION%\make"
echo STAN=$(CMDSTAN)/stan/>".bridgestan\bridgestan-%BRIDGESTAN_VERSION%\make\local"
echo %STAN%

echo [PROCESS] Installing Python modules (%TIME% %DATE%)
@REM Extract and install CondorCMF
powershell Expand-Archive condorcmf.zip -DestinationPath condorcmf
del condorcmf.zip
python -m pip install -q ./condorcmf

powershell Expand-Archive condorsmc.zip -DestinationPath condorsmc
del condorsmc.zip
python -m pip install -q ./condorsmc


echo --------------------------------------
echo [[ DEBUGGING ]]
echo --------------------------------------

dir

echo ---------------------------------------

cd .bridgestan
dir
cd ..

echo ---------------------------------------

g++ --version
make --version

echo ---------------------------------------
echo ---------------------------------------


echo [PROCESS] Launching CondorSMC (%TIME% %DATE%)
python -m condorsmc ^
    --session-id %session_id% ^
    --node-id %node_id% ^
    --mode %mode% ^
    --role %role% ^
    --coordinator-runtime %coordinator_runtime% ^
    --manager-runtime %manager_runtime% ^
    --follower-runtime %follower_runtime% ^
    --nfollowers %nfollowers% ^
    --model-dir %model_dir% ^
    --model %model% ^
    --nsamples %nsamples% ^
    --niters %niters% ^
    --proposal %proposal% ^
    --lkernel %lkernel% ^
    --recycling %recycling% ^
    --integrator %integrator% ^
    --step-size %step_size% ^
    --hmc-steps %hmc_steps% ^
    --seed %seed% ^
    --verbose

echo [PROCESS] Deactivate Conda environment (%TIME% %DATE%)
cd ./stan_env/Scripts
call deactivate
cd ../../
