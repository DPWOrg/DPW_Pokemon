<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "poke";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("fail: ". $conn->connect_error);
}


$pokemonName = urldecode($_GET['name']);

$stmt = $conn->prepare("SELECT * FROM pokemon WHERE name =?");
$stmt->bind_param("s", $pokemonName);
$stmt->execute();
$result = $stmt->get_result();

$pokemonData = null;
if ($result->num_rows > 0) {
    $pokemonData = $result->fetch_assoc();
}

$stmt->close();
$conn->close();
?>
<!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo htmlspecialchars($pokemonName); ?> Detailed information</title>
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
            background-color: #FFFFE0;
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
            background-color: #FFD700;
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
        }

       .pokemon-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-column-gap: 0;
            text-align: left;
            width: 80%; 
            margin: 0 auto; 
        }

       .pokemon-info table {
            width: 100%;
            border-collapse: collapse;
        }

       .pokemon-info th,
       .pokemon-info td {
            border: 1px solid #ccc;
            padding: 10px;
        }

       .pokemon-image {
            text-align: center;
            margin-bottom: 20px;
        }

       .pokemon-image img {
            max-width: 200px;
            height: auto;
        }
    </style>
</head>

<body>
    <h1><?php echo htmlspecialchars($pokemonName); ?> Detailed information</h1>
    <div class="main-container">
        <div class="sidebar">
             <a href="shouye.html">🏠 Home</a>
            <a href="tujian.php">📘 Pokédex</a>
            <a href="peidui.php">🤖 Intelligent Team Building</a>
        </div>
        <div class="main-content">
            <div class="pokemon-image">
                <img src="tupian/<?php echo htmlspecialchars($pokemonName); ?>.jpg" alt="<?php echo htmlspecialchars($pokemonName); ?>" onerror="this.style.display='none'">
            </div>
            <div class="pokemon-info">
                <?php if ($pokemonData!== null):
                    $keys = array_keys($pokemonData);
                    $half = ceil(count($keys) / 2);
                ?>
                    <table>
                        <?php for ($i = 0; $i < $half; $i++):
                            $key = $keys[$i];
                            $value = $pokemonData[$key];
                            if ($value === '') {
                                $value = 'null';
                            }
                        ?>
                            <tr>
                                <th><?php echo htmlspecialchars($key); ?></th>
                                <td><?php echo htmlspecialchars($value); ?></td>
                            </tr>
                        <?php endfor; ?>
                    </table>
                    <table>
                        <?php for ($i = $half; $i < count($keys); $i++):
                            $key = $keys[$i];
                            $value = $pokemonData[$key];
                            if ($value === '') {
                                $value = 'null';
                            }
                        ?>
                            <tr>
                                <th><?php echo htmlspecialchars($key); ?></th>
                                <td><?php echo htmlspecialchars($value); ?></td>
                            </tr>
                        <?php endfor; ?>
                    </table>
                <?php else: ?>
                    <p>No details were found for this Pokémon.</p>
                <?php endif; ?>
            </div>
        </div>
    </div>
</body>

</html>    