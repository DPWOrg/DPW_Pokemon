<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "poke";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: ". $conn->connect_error);
}

$search = isset($_GET['search']) ? $_GET['search'] : '';
$selectedType = isset($_GET['type']) ? strtolower($_GET['type']) : ''; 
$page = isset($_GET['page']) && is_numeric($_GET['page']) && $_GET['page'] > 0? $_GET['page'] : 1;
$limit = 40; 
$offset = ($page - 1) * $limit;

$sql = "SELECT pokedex_number, name, type1, type2 FROM pokemon";
$result = $conn->query($sql);

$pokemonData = [];
if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $pokemonData[] = $row;
    }
}

$filteredPokemon = [];
foreach ($pokemonData as $pokemon) {
    $nameMatch = empty($search) || strpos(strtolower($pokemon['name']), strtolower($search))!== false;
    $typeMatch = empty($selectedType) || $selectedType === 'all' || ($pokemon['type1'] === $selectedType || $pokemon['type2'] === $selectedType);

    if ($nameMatch && $typeMatch) {
        $filteredPokemon[] = $pokemon;
    }
}


$totalPokemon = count($filteredPokemon);
$totalPages = ceil($totalPokemon / $limit);
$filteredPokemon = array_slice($filteredPokemon, $offset, $limit);

$conn->close();
?>
<!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>宝可梦图鉴</title>
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
       .search-container {
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

       .search-container h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 20px;
        }

       .search-container form {
            display: flex;
            align-items: center;
            justify-content: center;
        }

       .search-container input[type="text"] {
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 3px;
            margin-right: 5px;
            width: 300px;
        }

       .search-container button {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background-color: #e0e0e0;
            color: #333;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

       .search-container button:hover {
            background-color: #FFD700;
            transform: scale(1.05);
        }
       .filter-container {
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

       .filter-container h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 20px;
        }

       .filter-container a {
            margin-right: 8px;
            margin-bottom: 8px;
            display: inline-block;
        }

       .filter-container button {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background-color: #e0e0e0;
            color: #333;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

       .filter-container button:hover {
            background-color: #FFD700;
            transform: scale(1.05);
        }

       .pokemon-list {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }

       .pokemon-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }

       .pokemon-card:hover {
            transform: scale(1.05);
        }

       .pokemon-card img {
            width: 100%;
            height: auto;
            margin-bottom: 10px;
        }

       .pokemon-attributes {
            font-size: 14px;
            color: #666;
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
    </style>
</head>

<body>
    <main class="main-container">
        <div class="sidebar">
             <a href="shouye.html">🏠 Home</a>
            <a href="tujian.php">📘 Pokédex</a>
            <a href="peidui.php">🤖 Intelligent Team Building</a>
        </div>
        <section class="main-content">
            <div class="section-wrapper">
                <h2 class="section-title">📘 Pokémon Illustrated</h2>
                <div class="search-container">
                    <form action="tujian.php" method="get">
                        <input type="text" name="search" placeholder="Search for Pokémon Names" value="<?php echo htmlspecialchars($search); ?>">
                        <button type="submit">Search</button>
                        <input type="hidden" name="type" value="<?php echo htmlspecialchars($selectedType); ?>">
                        <input type="hidden" name="page" value="<?php echo $page; ?>">
                    </form>
                    <p>(ps: the uncollected pictures are temporarily replaced by mimichu's pictures)</p>
                </div>
                <div class="filter-container">
                    <a href="tujian.php?search=<?php echo htmlspecialchars($search);?>&type=all&page=<?php echo $page;?>">
                        <button>all</button>
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
                        <a href="tujian.php?search=<?php echo htmlspecialchars($search);?>&type=<?php echo htmlspecialchars(strtolower($type));?>&page=<?php echo $page;?>">
                            <button><?php echo $type;?></button>
                        </a>
                    <?php endforeach; ?>
                </div>
                <div class="pokemon-list">
                <?php foreach ($filteredPokemon as $pokemon):?>
                    <div class="pokemon-card" onclick="window.location.href='xiangqing.php?name=<?php echo urlencode($pokemon['name']); ?>'">
                        <h2><span class="pokedex-number"><?php echo $pokemon['pokedex_number'];?></span> </h2>
                    <?php 
                        $imagePath = 'tupian/' . htmlspecialchars($pokemon['name']) . '.jpg';
                        $displayImage = file_exists($imagePath) ? $imagePath : 'tupian/Mimikyu.jpg';
                    ?>
                    <img src="<?php echo $displayImage; ?>" alt="<?php echo htmlspecialchars($pokemon['name']); ?>">
                    <h3><?php echo htmlspecialchars($pokemon['name']);?></h3>
                    <div class="pokemon-attributes">
                        Attribute：<?php echo $pokemon['type1']; if (!empty($pokemon['type2'])) { echo ", ". $pokemon['type2']; }?>
                        </div>
                    </div>
                    <?php endforeach;?>
                </div>
                <div class="pagination">
                    <?php for ($i = 1; $i <= $totalPages; $i++): ?>
                        <a href="tujian.php?search=<?php echo htmlspecialchars($search);?>&type=<?php echo htmlspecialchars($selectedType);?>&page=<?php echo $i;?>" <?php if ($i == $page) echo 'class="active"';?>>
                            <?php echo $i;?>
                        </a>
                    <?php endfor; ?>
                </div>
            </div>
        </section>
    </main>
</body>

</html>    