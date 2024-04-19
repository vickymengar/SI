from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/obtener_prediccion', methods=['POST'])
def obtener_prediccion():
    # Obtener los datos del formulario
    delito = request.form['delito']
    mes = request.form['mes']

    # Ejecutar el script de predicciones
    resultado = subprocess.check_output(['python', 'Predicciones.py', delito, mes])
    resultado_decodificado = resultado.decode('utf-8')

    return jsonify({'resultado': resultado_decodificado})

if __name__ == '__main__':
    app.run(debug=True)
