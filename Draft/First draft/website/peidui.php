<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "poke";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: ". $conn->connect_error);
}

$sql = "SELECT pokedex_number, name, type1, type2 FROM pokemon";
$result = $conn->query($sql);

$pokemonData = [];
if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $pokemonData[] = $row;
    }
}

$selectedType = isset($_GET['type']) ? $_GET['type'] : '';
$searchTerm = isset($_GET['search']) ? $_GET['search'] : '';
$page = isset($_GET['page']) && is_numeric($_GET['page']) && $_GET['page'] > 0? $_GET['page'] : 1;
$limit = 40; 
$offset = ($page - 1) * $limit;

$filteredPokemon = [];
foreach ($pokemonData as $pokemon) {
    $nameMatch = empty($searchTerm) || strpos(strtolower($pokemon['name']), strtolower($searchTerm))!== false;
    $typeMatch = empty($selectedType) || $selectedType === 'All' || ($pokemon['type1'] === $selectedType || $pokemon['type2'] === $selectedType);

    if ($nameMatch && $typeMatch) {
        $filteredPokemon[] = $pokemon;
    }
}

$totalPokemon = count($filteredPokemon);
$totalPages = ceil($totalPokemon / $limit);
$filteredPokemon = array_slice($filteredPokemon, $offset, $limit);

$selectedPokemon = isset($_GET['selected']) ? explode(',', $_GET['selected']) : [];

$conn->close();
?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>peidui</title>
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
            border: 3px solid #FFD700;
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
            border-bottom: 2px solid #FFD700;
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
            border-bottom: 2px solid #FFD700;
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
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: solid #FFD700;
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
            background-color: #FFD700;
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
    </style>
</head>

<body>
    <header>
        <h1 class="page-title">🎮 Pokémon Intelligent Teaming Platform</h1>
    </header>
    <main class="main-container">
        <div class="sidebar">
            <a href="shouye.html">🏠 Home</a>
            <a href="tujian.php">📘 Pokédex</a>
            <a href="peidui.php">🤖 Intelligent Team Building</a>
        </div>
        <section class="main-content">
            <div class="section-wrapper">
                <div class="function-intro">
                    <p>Introduction to Modes and Functions</p>
                    <p>(ps: the uncollected pictures are temporarily replaced by mimichu's pictures)</p>
                </div>
                <div class="function-implementation">
                    <div class="selected-pokemon">
                        <h3>Selected Pokémon</h3>
                        <?php if (!empty($selectedPokemon) && count($selectedPokemon) <= 7): ?>
                            <div class="selected-pokemon-row">
                                <?php foreach ($selectedPokemon as $pokedexNumber):
                                    foreach ($pokemonData as $pokemon) {
                                        if ($pokemon['pokedex_number'] == $pokedexNumber) {
                                            $imagePath = 'tupian/' . htmlspecialchars($pokemon['name']) . '.jpg';
                                            $displayImage = file_exists($imagePath) ? $imagePath : 'tupian/Mimikyu.jpg';
                                            echo '<div class="selected-pokemon-item">';
                                            echo "<span>{$pokemon['pokedex_number']}. {$pokemon['name']}</span>";
                                            echo "<div class='pokemon-attributes'>attribute:{$pokemon['type1']}";
                                            if (!empty($pokemon['type2'])) {
                                                echo ", ". $pokemon['type2'];
                                            }
                                            echo "</div>";
                                            echo "<img src='{$displayImage}' alt='{$pokemon['name']}' class='pokemon-image'>";
                                            echo "<form action='' method='get'>";
                                            echo "<input type='hidden' name='search' value='{$searchTerm}'>";
                                            echo "<input type='hidden' name='type' value='{$selectedType}'>";
                                            echo "<input type='hidden' name='page' value='{$page}'>";
                                            $currentSelected = array_diff($selectedPokemon, [$pokedexNumber]);
                                            echo "<input type='hidden' name='selected' value='". implode(',', $currentSelected). "'>";
                                            echo "<button type='submit'>Cancel</button>";
                                            echo "</form>";
                                            echo '</div>';
                                            break;
                                        }
                                    }
                                endforeach; ?>
                            </div>
                        <?php elseif (!empty($selectedPokemon)): ?>
                            <p style="color: red;">The maximum number of Pokémon already selected is six; new selections are not counted.</p>
                        <?php else: ?>
                            <p>No choice of pokemon</p>
                        <?php endif; ?>
                    </div>
                    <div class="type-filter">
                        <h3>Select by Type</h3>
                        <a href="?search=<?php echo htmlspecialchars($searchTerm);?>&type=All&page=<?php echo $page;?>&selected=<?php echo implode(',', $selectedPokemon);?>">
                            <button>All</button>
                        </a>
                        <?php
                        $types = [];
                        foreach ($pokemonData as $pokemon) {
                            if (!in_array($pokemon['type1'], $types)) {
                                $types[] = $pokemon['type1'];
                            }
                            if ($pokemon['type2'] &&!in_array($pokemon['type2'], $types)) {
                                $types[] = $pokemon['type2'];
                            }
                        }
                        foreach ($types as $type):
                        ?>
                            <a href="?search=<?php echo htmlspecialchars($searchTerm);?>&type=<?php echo htmlspecialchars($type);?>&page=<?php echo $page;?>&selected=<?php echo implode(',', $selectedPokemon);?>">
                                <button><?php echo $type;?></button>
                            </a>
                        <?php endforeach; ?>
                    </div>
                    <div class="pokemon-selection">
                        <h3>select</h3>
                        <form action="" method="get">
                            <input type="text" name="search" placeholder="Search for Names" value="<?php echo htmlspecialchars($searchTerm); ?>">
                            <input type="hidden" name="type" value="<?php echo htmlspecialchars($selectedType); ?>">
                            <input type="hidden" name="page" value="<?php echo $page; ?>">
                            <input type="hidden" name="selected" value="<?php echo implode(',', $selectedPokemon); ?>">
                        </form>
                        <br><br>
                        <div id="pokemon-list" class="pokemon-row">
                            <?php foreach ($filteredPokemon as $pokemon):
                                if (!in_array($pokemon['pokedex_number'], $selectedPokemon)):
                                    $imagePath = 'tupian/' . htmlspecialchars($pokemon['name']) . '.jpg';
                                    $displayImage = file_exists($imagePath) ? $imagePath : 'tupian/Mimikyu.jpg';
                            ?>
                                <div class="pokemon-item">
                                    <a href="#" onclick="return checkSelection('<?php echo $pokemon['pokedex_number'];?>', '<?php echo implode(',', $selectedPokemon);?>')">
                                        <span><?php echo "{$pokemon['pokedex_number']}. {$pokemon['name']}";?></span>
                                        <div class="pokemon-attributes">
                                            Properties:<?php echo $pokemon['type1']; if (!empty($pokemon['type2'])) { echo ", ". $pokemon['type2']; }?>
                                        </div>
                                        <img src="<?php echo $displayImage; ?>" alt="<?php echo $pokemon['name']; ?>" class="pokemon-image">
                                    </a>
                                </div>
                            <?php endif; endforeach; ?>
                        </div>
                        <div class="pagination">
                            <?php for ($i = 1; $i <= $totalPages; $i++): ?>
                                <a href="?search=<?php echo htmlspecialchars($searchTerm);?>&type=<?php echo htmlspecialchars($selectedType);?>&page=<?php echo $i;?>&selected=<?php echo implode(',', $selectedPokemon);?>" <?php if ($i == $page) echo 'class="active"';?>>
                                    <?php echo $i;?>
                                </a>
                            <?php endfor; ?>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        <script>
            function checkSelection(pokemonId, currentSelected) {
                var selectedArray = currentSelected.split(',');
                if (selectedArray.length > 6) {
                    alert('The maximum number of sprites selected is six, and no new sprites can be added.');
                    return false;
                } else {
                    var url = '?search=<?php echo htmlspecialchars($searchTerm);?>&type=<?php echo htmlspecialchars($selectedType);?>&page=<?php echo $page;?>&selected=' + currentSelected + ',' + pokemonId;
                    window.location.href = url;
                    return true;
                }
            }
        </script>
    </main>
</body>

</html>