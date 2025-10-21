$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "$scriptPath\my_strategy.py"

py -3 $algoPath
