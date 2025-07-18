## DPW Pokémon – Machine Learning Data Analysis Project

### Project Overview

The DPW Pokémon project is an open‑source analysis initiative that applies machine‑learning techniques to the official Pokémon dataset. Using the Pokémon universe as a backdrop, we explore each creature’s attributes and combat strength through data‑science methods. Whether you are a Pokémon enthusiast or a data‑science beginner, this project helps you uncover interesting patterns and insights hidden in Pokémon data. The project relies on the official Kaggle Pokémon dataset, which contains detailed information on hundreds of Pokémon (nearly 40 features, including types, base stats, and type match‑ups). By combining visualization with machine‑learning models, we reveal how attributes are distributed, how battle power varies, and how AI algorithms can assemble an ideal battle team.

### Type and Attribute Analysis

We first explored and visualized the core attributes of each Pokémon. In the dataset every Pokémon has one or two types (e.g., Fire, Water) plus six base stats—HP, Attack, Defense, Special Attack, Special Defense, and Speed. Statistical analysis allowed us to plot the distribution of primary and secondary types. Water‑type Pokémon are the most numerous overall, while Flying is the most common secondary type (473 Pokémon have Flying as a secondary type). As for dual‑type pairs, **Normal + Flying** occurs most frequently (87 Pokémon share this combination).
We also compared average base stats across types and found that Dragon‑type Pokémon generally have higher average battle power, whereas Normal‑type Pokémon are more balanced across all stats. Legendary Pokémon stand out: their average total base stat is about 615, compared with approximately 410 for non‑legendary Pokémon—matching player intuition. Besides summary tables, we supply relational charts illustrating interactions among stats—for example, Speed vs. Attack or Weight vs. HP—so users can grasp attribute interactions at a glance.

### Machine Learning Models

After understanding the data, we applied several machine‑learning techniques to probe further into the Pokémon world:

* **Classification** models predict certain categories (e.g., whether a Pokémon is legendary) based on base stats.
* **Clustering** groups Pokémon with similar attributes, revealing hidden clusters of comparable creatures.
* **Team strength evaluation** leverages a **Graph Neural Network (GNN)** to score a team of six Pokémon. We employ a Graph Attention Network (GAT) that encodes each team: every Pokémon node carries 114 features (its six base stats, type information, and type‑match multipliers). Two GAT convolution layers plus a dense head output an overall team strength score in the range 0–1.
* **Genetic Algorithm (GA)** optimization refines team composition. Starting with 500 random teams, we evolve them using an adaptive mutation rate of roughly 15 %. A custom “type‑aware” crossover operator preserves complementary type combinations from parents. The fitness function blends type‑advantage score, average base stats, and the GNN prediction, weighted 0.4 : 0.3 : 0.3. Through selection, crossover, and mutation, the GA identifies teams with high combat power and strong type synergy.

### Data Source and Preprocessing

Data come from the Pokémon dataset on Kaggle, covering Generations I–VII. It includes 18 elemental types, detailed base stats (HP, Attack, Defense, Special Attack, Special Defense, Speed), and type‑effectiveness multipliers against all 18 types. The raw dataset holds about 800 Pokémon and 40+ columns.
During preprocessing we cleaned missing or anomalous values: Pokémon without gender ratios (genderless Pokémon) received a special tag, and single‑type Pokémon had their missing second type filled with “None.” All numeric features were standardized (Z‑score) to remove scale differences.
For categorical features we engineered **one‑hot encodings** of primary and secondary types and retained the numeric type‑match multipliers for modeling type advantages. Fields unrelated to battle (e.g., Pokédex ID, egg steps) were dropped, and derived metrics such as a “BMI” (from height and weight) were added. After cleaning and feature engineering, we reduced the feature set from 41 to about 17 key features, yielding a streamlined modeling dataset.

### Database Backend

A **relational database** manages and queries Pokémon data. We chose MySQL (via the XAMPP stack) as the back‑end and imported the cleaned dataset into a local database named “poke.” A single Pokémon table stores each creature’s full details—name, types, base stats, type‑effectiveness values, and more.
In the application layer we connect to MySQL through SQLAlchemy ORM in Python so that machine‑learning models can easily fetch data. The website front‑end uses PHP scripts to query MySQL, enabling search and display of Pokémon data. For instance, the “Pokédex” page filters Pokémon by name or type after querying the database.
The full system runs on a local **LAMP** stack: Apache + PHP serves front‑end pages and APIs, MySQL stores data, and Python performs machine‑learning computations. When a user clicks “Run Python Script,” the back end launches our ML modules (e.g., `Test2.py` / `Test3.py`), reads data from MySQL, executes the team‑optimization algorithm, and writes results back for the web page to display. This layered architecture decouples storage from computation, combining efficient SQL queries with Python‑based AI methods.

### Usage Guide and Dependencies

To run the project you need the following environment and libraries:

* **Python 3** plus core packages: `json`, `random`, `os`, `NumPy`, `Pandas`, `PyTorch`, `PyTorch‑Geometric`, `SQLAlchemy`, `tqdm`, etc. (be sure to install `torch‑geometric` and its dependencies for the GNN).
* **MySQL** (XAMPP recommended) for data storage and an **Apache/PHP** server for the front‑end interface.
* **Pokémon dataset**: download the official Kaggle CSV, or use the provided `poke.sql` dump to load data into MySQL.

Configuration and execution steps:

1. **Database setup**

   * Start MySQL, create a local database named “poke,” then import the Kaggle CSV (or run `poke.sql` to create the table and insert data).
   * Ensure necessary indexes (types, etc.) are in place; we have indexed the type fields for faster queries.

2. **Deploy the front end**

   * Start Apache (XAMPP can launch Apache and MySQL together).
   * Copy the project’s web files (e.g., `shouye.html`, `tujian.php`, `peidui.php`) into Apache’s `htdocs` directory.
   * You can now access the interface via your browser.

3. **Run the application**

   * Open `http://localhost/shouye.html` to reach the home page, which links to sample charts showing relationships among attributes.
   * Use the left‑hand navigation to enter the **Pokédex** or **Team Builder**:

     * **Pokédex** – search by name or filter by type. Clicking an avatar shows full details (types, stats, description).
     * **Team Builder** – search and click Pokémon to add them to a candidate list. After selecting six Pokémon, a “Submit Team” button appears. Press it, then choose “View Team Details.”

       * If the recommended Pokémon list (Rx) does not appear, click **Run Python Script** to launch the back‑end optimizer. The Python script (`Test2.py`, etc.) runs the GA and GNN, taking around ten minutes. When the browser alerts “Python script executed successfully,” the best team and simulated battle stats display at the top.
       * **Note**: If clicking **Run Python Script** does nothing, edit `run_python.py` and set `$pythonPath` to your local Python executable.

4. **Command‑line execution (optional)**

   * Advanced users can run Python scripts directly—for example, `python Final_version/Test3.py` triggers the full optimization flow and prints progress. Before doing so, confirm the database connection inside the script (default is local `pokemon`) and ensure all dependencies are installed.

### Open‑Source License

Project code is released under the MIT License. Pokémon data, images, and related assets remain the property of The Pokémon Company; our use is strictly for non‑commercial research and analysis.


### Originator：

Qiao Yichang 2330026132

Ma Xinyao 2330026119

Huo Yuxin 2330026064

Tang Jianxu 2330026143

Liu Chenrui 2330004018 

Wang Zhengyang 2330031281
