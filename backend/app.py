from flask import Flask
from flask_cors import CORS # type: ignore
from app.routes import main
from app.serial_handler import ArduinoSerialHandler

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize Arduino handler
    arduino = ArduinoSerialHandler(port='COM3')  # Adjust COM port as needed
    
    # Register blueprints
    app.register_blueprint(main)
    
    # Add Arduino routes
    @app.route('/arduino/connect', methods=['POST'])
    def connect_arduino():
        if arduino.connect():
            return jsonify({"status": "success", "message": "Connected to Arduino"})
        return jsonify({"status": "error", "message": "Failed to connect to Arduino"}), 500

    @app.route('/arduino/start', methods=['POST'])
    def start_monitoring():
        if arduino.start_monitoring():
            return jsonify({"status": "success", "message": "Started Arduino monitoring"})
        return jsonify({"status": "error", "message": "Failed to start monitoring"}), 500

    @app.route('/arduino/stop', methods=['POST'])
    def stop_monitoring():
        arduino.stop_monitoring()
        return jsonify({"status": "success", "message": "Stopped Arduino monitoring"})

    @app.route('/arduino/data', methods=['GET'])
    def get_arduino_data():
        return jsonify(arduino.get_latest_data())

    return app

    

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)