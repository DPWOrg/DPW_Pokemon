<?php
header('Content-Type: application/json');

$isWindows = strtoupper(substr(PHP_OS, 0, 3)) === 'WIN';
if ($isWindows) {
    exec('where python', $pythonPaths);
} else {
    exec('which python', $pythonPaths);
}

if (empty($pythonPaths)) {
    $response = [
        'status' => 'error',
        'output' => ['Python not found in the system PATH.']
    ];
} else {
    $pythonPath = $pythonPaths[0];
    $scriptPath = 'GAN_Gnentic_Analysis.py'; // Casematters


    // Allow the capture of stderr
    $command = "$pythonPath $scriptPath 2>&1";
    exec($command, $output, $returnCode);

    if ($returnCode === 0) {
        $response = [
            'status' => 'success',
            'output' => $output
        ];
    } else {
        $response = [
            'status' => 'error',
            'output' => $output
        ];
    }
}

echo json_encode($response);
?>
