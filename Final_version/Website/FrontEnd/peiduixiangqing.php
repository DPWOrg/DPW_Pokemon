<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "poke";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: ". $conn->connect_error);
}

$sqlBattleRecords = "SELECT * FROM battle_records";
$resultBattleRecords = $conn->query($sqlBattleRecords);

$battleRecords = [];
if ($resultBattleRecords->num_rows > 0) {
    while ($row = $resultBattleRecords->fetch_assoc()) {
        $battleRecords[] = $row;
    }
    // Sort the data in reverse order
    $battleRecords = array_reverse($battleRecords);
}

function getPokemonName($conn, $pokedexNumber) {
    if ($pokedexNumber === null) {
        return '';
    }
    $sql = "SELECT name FROM pokemon WHERE pokedex_number =?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("s", $pokedexNumber);
    $stmt->execute();
    $result = $stmt->get_result();
    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        return $row['name'];
    }
    return "Unknown";
}

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['delete_id'])) {
    $deleteId = $_POST['delete_id'];
    $sqlDelete = "DELETE FROM battle_records WHERE id =?";
    $stmtDelete = $conn->prepare($sqlDelete);
    $stmtDelete->bind_param("i", $deleteId);
    if ($stmtDelete->execute()) {
        header("Location: {$_SERVER['PHP_SELF']}");
        exit;
    } else {
        echo "Error deleting record: ". $conn->error;
    }
    $stmtDelete->close();
}
?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pokémon Battle Records</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
            position: relative;
        }

        body::before {
            content: "";
            background-image: url('back.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0.8;
            z-index: -1;
        }

        header {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 1vh;
            position: relative;
        }

       .page-title {
            text-align: center;
            color: #333;
            padding: 20px;
            margin: 10px 0;
            border: 3px solid #FFFF99;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: rgba(255, 255, 255, 0.8);
        }

       .main-container {
            display: flex;
            width: 90%;
            max-width: 1200px;
        }

       .sidebar {
            position: fixed;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: fit-content;
        }

       .sidebar a {
            display: block;
            color: #333;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
            transition: background-color 0.3s ease;
        }

       .sidebar a:hover {
            background-color: #FFFF99;
            color: white;
        }

       .main-content {
            flex: 1;
            padding: 20px;
            text-align: center;
            color: #333;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
        }

       .news-section {
            margin-top: 20px;
        }

       .news-section ul {
            list-style-type: none;
            padding: 0;
        }

       .news-section li {
            margin-bottom: 5px;
        }

       .news-section a {
            color: #007BFF;
            text-align: center;
            text-decoration: none;
        }

       .news-section a:hover {
            text-decoration: underline;
        }

       .content-section {
            margin-bottom: 30px;
            text-align: left;
        }

       .content-section h2 {
            border-bottom: 2px solid #FFFF99;
            padding-bottom: 10px;
        }

       .content-section p {
            line-height: 1.6;
        }

       .section-wrapper {
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 20px;
            background-color: rgba(255, 250, 240, 0.8);
            border-radius: 10px;
            padding: 20px;
        }

       .section-title {
            text-align: center;
            font-size: 28px;
            color: #333;
            border-bottom: 2px solid #FFFF99;
            padding-bottom: 10px;
        }

       .function-intro,
       .function-implementation {
            text-align: left;
            line-height: 1.6;
            padding: 0 20px;
        }

       .selected-pokemon {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }

       .selected-pokemon-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }

       .selected-pokemon-item {
            width: calc(16.66% - 10px);
            margin-bottom: 10px;
            text-align: center;
        }

       .pokemon-selection {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }

       .type-filter {
            border: 2px solid #FFFF99;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: solid #FFFF99;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

       .type-filter h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 20px;
        }

       .type-filter a {
            margin-right: 8px;
            margin-bottom: 8px;
            display: inline-block;
        }

       .type-filter button {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background-color: #e0e0e0;
            color: #333;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

       .type-filter button:hover {
            background-color: #FFFF99;
            transform: scale(1.05);
        }

       #pokemon-list div {
            cursor: pointer;
            padding: 5px;
            border-radius: 3px;
        }

       #pokemon-list div:hover {
            background-color: #f0f0f0;
        }

       #type-buttons button {
            margin-right: 5px;
            padding: 5px 10px;
            border: none;
            border-radius: 3px;
            background-color: #e0e0e0;
            cursor: pointer;
        }

       #type-buttons button:hover {
            background-color: #d0d0d0;
        }

       .pokemon-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }

       .pokemon-item {
            width: calc(20% - 10px);
            margin-bottom: 10px;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            transition: transform 0.3s ease;
        }

       .pokemon-item:hover {
            transform: scale(1.05);
        }

       .pokemon-attributes {
            font-size: 14px;
            color: #666;
        }

       .pokemon-image {
            width: 100px;
            height: 100px;
            object-fit: cover;
            margin-top: 5px;
        }

       .pagination a {
            display: inline-block;
            padding: 5px 10px;
            margin: 0 5px;
            background-color: #fffaf0;
            border-radius: 3px;
            text-decoration: none;
            color: #8b4513;
        }

       .pagination a:hover {
            background-color: #ffdab9;
        }

       .pagination a.active {
            background-color: #ffdab9;
            color: white;
        }

       .pokemon-selection form {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }

       .pokemon-selection form input[type="text"] {
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 3px;
            margin-right: 5px;
            width: 60%;
        }

       .pokemon-selection form input[type="hidden"] {
            display: none;
        }

       .selected-pokemon form {
            display: flex;
            justify-content: center;
            margin-top: 15px;
        }

       .selected-pokemon form button[type="submit"] {
            padding: 12px 24px;
            background-color: #FFFF99;
            color: #333;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

       .selected-pokemon form button[type="submit"]:hover {
            background-color: #FFFFCC;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }

       .selected-pokemon form button[type="submit"]:active {
            background-color: #FFFF66;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transform: translateY(1px);
        }

       .battle-records-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

       .battle-records-table th,
       .battle-records-table td {
            border: 1px solid #ccc;
            padding: 8px;
            text-align: center;
        }

       .battle-records-table th {
            background-color: #f0f0f0;
        }

       .delete-button {
            background-color: #FFC107;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 6px 12px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

       .delete-button:hover {
            background-color: #FFC107;
        }

       .load-button {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #FFC107;
            color: #333;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

       .load-button:hover {
            background-color: #FFFFCC;
        }

        #loading {
            display: none;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        #result {
            margin-top: 20px;
        }

        #runScript {
            padding: 12px 24px;
            background-color: #FFC107;
            color: #333;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        #runScript:hover {
            background-color: #FFFFCC;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }

        #runScript:active {
            background-color: #FFC107;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transform: translateY(1px);
        }
    </style>
</head>

<body>
    <header>
        <h1 class="page-title">Pokémon Battle Records</h1>
    </header>
    <main class="main-container">
        <div class="sidebar">
            <a href="shouye.html">🏠 Home</a>
            <a href="tujian.php">📘 Pokédex</a>
            <a href="peidui.php">🤖 Intelligent Team Building</a>
        </div>
        <section class="main-content">
            <button id="runScript">Run Python Script</button>
            <div id="loading">Running Python script...</div>
            <div id="result"></div>

            <script>
                document.getElementById('runScript').addEventListener('click', function () {
                    document.getElementById('loading').style.display = 'block';

                    var xhr = new XMLHttpRequest();
                    xhr.open('GET', 'run_python.php', true);
                    xhr.onreadystatechange = function () {
                        if (xhr.readyState === 4) {
                            document.getElementById('loading').style.display = 'none';
                            if (xhr.status === 200) {
                                var data = JSON.parse(xhr.responseText);
                                var resultDiv = document.getElementById('result');
                                if (data.status === 'success') {
                                    resultDiv.innerHTML = '<p>Python script executed successfully!</p>';
                                    data.output.forEach(function (line) {
                                        resultDiv.innerHTML += '<p>' + line + '</p>';
                                    });
                                } else {
                                    resultDiv.innerHTML = '<p>Error executing Python script:</p>';
                                    data.output.forEach(function (line) {
                                        resultDiv.innerHTML += '<p>' + line + '</p>';
                                    });
                                }
                            } else {
                                var resultDiv = document.getElementById('result');
                                resultDiv.innerHTML = '<p>Error: ' + xhr.statusText + '</p>';
                            }
                        }
                    };
                    xhr.send();
                });
            </script>
            <div class="section-wrapper">
                <div class="function-intro">
                    <p>method: After selecting six Pokémon, click "Submit". Then click the link "View team details" to enter the details page. After the selected Pokémon (S) are successfully inserted, click "Run Python Script" and wait for about ten minutes. After "Python script executed successfully!" appears, refresh the page. R represents the team recommended for the current lineup just now</p>
                </div>
                <div class="function-implementation">
                    <?php if (!empty($battleRecords)): ?>
                        <table class="battle-records-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>S1</th>
                                    <th>S2</th>
                                    <th>S3</th>
                                    <th>S4</th>
                                    <th>S5</th>
                                    <th>S6</th>
                                    <th>R1</th>
                                    <th>R2</th>
                                    <th>R3</th>
                                    <th>R4</th>
                                    <th>R5</th>
                                    <th>R6</th>
                                    <th>Control</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php foreach ($battleRecords as $record): ?>
                                    <tr>
                                        <td><?php echo $record['id']; ?></td>
                                        <td><?php echo getPokemonName($conn, $record['selected_pokemon_1']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['selected_pokemon_2']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['selected_pokemon_3']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['selected_pokemon_4']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['selected_pokemon_5']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['selected_pokemon_6']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['recommended_pokemon_1']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['recommended_pokemon_2']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['recommended_pokemon_3']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['recommended_pokemon_4']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['recommended_pokemon_5']); ?></td>
                                        <td><?php echo getPokemonName($conn, $record['recommended_pokemon_6']); ?></td>
                                        <td>
                                            <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
                                                <input type="hidden" name="delete_id" value="<?php echo $record['id']; ?>">
                                                <button type="submit" class="delete-button">Delete</button>
                                            </form>
                                        </td>
                                    </tr>
                                <?php endforeach; ?>
                            </tbody>
                        </table>
                    <?php else: ?>
                        <p>No battle records found.</p>
                    <?php endif; ?>
                </div>
            </div>
        </section>
    </main>
    <?php $conn->close(); ?>
</body>

</html>    