from flask import Flask, render_template, request, jsonify

sudoku_app = Flask(__name__)


# Define the initial Sudoku grid (0 represents empty cells)
sudoku_grid = [
    [5, 3, 0],
    [6, 0, 0],
    [0, 9, 8]
]

# Define the solution for the Sudoku grid
sudoku_solution = [
    [5, 3, 4],
    [6, 7, 2],
    [1, 9, 8]
]

@sudoku_app.route("/sudoku")

def sudoku():
    return render_template("sudoku_game.html", grid=sudoku_grid)

@sudoku_app.route("/validate", methods=["POST"])
def validate():
    # Get the user input from the frontend
    data = request.json
    row = data["row"]
    col = data["col"]
    value = data["value"]

    # Check if the user's value matches the solution
    if sudoku_solution[row][col] == value:
        return jsonify({"result": "correct", "message": "Correct Number!"})
    else:
        return jsonify({"result": "wrong", "message": "Incorrect Number!"})

if __name__ == "__main__":
    sudoku_app.run(debug=True)
