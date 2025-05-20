First, open XAMPP and connect Apache and MySQL.

Create a database named "poke", and then import the "poke.sql" file from the folder.

Put all relevant files into the "htdocs" folder of the XAMPP directory.

Open your browser and enter "http://localhost/shouye.html" to access the homepage.

The two links at the bottom of the homepage show the relationships between individual Pokémon attribute values, which are created using Python.

The three links on the left lead to the "Homepage", "Pokedex", and "Team Building" pages respectively.

Pokedex Page
On the Pokedex page, you can find Pokémon by searching for their names or selecting their attributes. Simply click on a Pokémon's image to view its detailed information.

Team Building Page
The search mechanism on the Team Building page is similar to that of the Pokedex page. After clicking on a Pokémon, it will be added to the "Selected Pokémon" list. When you have selected six Pokémon, a submit button will appear. Click the submit button to enter the "Pending Status" (you cannot successfully submit a Pokémon team in this state).
Click the link "View Team Details" to enter the "Pokémon Battle Records" display page. 
If there are no Rx (Recommended Pokémon) in the top - most row, it means the task is "Pending". At this time, click the "Run Python Script" button, and the "test2.py" file will be invoked for team - building, which takes approximately ten minutes. When the message "Python script executed successfully" appears, it indicates that the team - building process is complete. The top - most row then shows the results of your team - building.

PS: If the "Run Python Script" button doesn't work after being clicked, it may be because the Python path cannot be found. You need to open the "run_python.py" file and change the corresponding path for "$pythonPath".

PS: This team - building method is not designed for traditional Pokémon battles. Instead, it is based on a 6 - on - 6 simultaneous battle mode we've constructed. Therefore, the team - building results may differ significantly from those of traditional battles