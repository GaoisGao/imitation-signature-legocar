'''
Easiet way to use:

import lelib
from lelib import singleMotor, doubleMotor, colorSensor, controller

'''

import time
import legoeducation as le

class singleMotor(le.SingleMotor):
    def __init__(self):
        super().__init__()

    def connect(self, card_serial, card_color=None):
        for attempt in range(5):
            try:
                super().connect(card_color=card_color, card_serial=card_serial)
                break
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                else:
                    raise
        if not self.connected:
            raise ConnectionError('Error connecting to Single Motor.')
            
        
    def spin(self, rotations=1):
        self.motor_run_for_degrees(rotations * 360)

    def stop(self):
        self.motor_stop()
    
    def set_speed(self, speed):
        self.motor_set_speed(speed)

    def run(self):
        self.motor_run() 


class doubleMotor(le.DoubleMotor):

    def connect(self, card_serial, card_color=None):
        for attempt in range(5):
            try:
                super().connect(card_color=card_color, card_serial=card_serial)
                break
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                else:
                    raise
        if not self.connected:
            raise ConnectionError('Error connecting to Double Motor.')
       
    def move_steps(self, step=1):
        '''
        Move both motors at once for given number of steps. 
        One step defined to be 180 degrees.
        '''
        self.movement_move_for_degrees(-180*step)

    def run(self):
        self.movement_move(direction=le.MOVEMENT_MOVE_DIRECTION_BACKWARD)

    def run_time(self, time=2000):
        self.movement_move_for_time(time)

    
    def run_left(self, degrees=None):
        if degrees is None:
            self.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT)
        else:
            self.motor_run_for_degrees(degrees=degrees, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT)

    def run_right(self, degrees=None):
        if degrees is None:
            self.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT)
        else:
            self.motor_run_for_degrees(degrees=degrees, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT)
    

    def turn_left(self, degrees=90):
        '''
        Turns left by specified number of degrees.
        '''
        self.movement_turn_for_degrees(degrees, direction=le.MOVEMENT_TURN_DIRECTION_LEFT)

    def turn_right(self, degrees=90):
        '''
        Turns right by specified number of degrees.
        '''
        self.movement_turn_for_degrees(degrees, direction=le.MOVEMENT_TURN_DIRECTION_RIGHT)

    def set_speed(self, speed):
        '''
        Set speed of both motors for individual rotation and movement.
        '''
        self.motor_set_speed(speed, motor=le.MOTOR_LEFT)   
        self.motor_set_speed(speed, motor=le.MOTOR_RIGHT)   
        self.movement_set_speed(speed)

    def set_speed_left(self, speed):
        '''
        Set speed of left motor for individual rotation.
        '''
        self.motor_set_speed(speed, motor=le.MOTOR_LEFT)   

    def set_speed_right(self, speed):
        '''
        Set speed of right motor for individual rotation.
        '''
        self.motor_set_speed(speed, motor=le.MOTOR_RIGHT)   


    def stop(self):
        self.motor_stop()


class controller(le.Controller):

    def connect(self, card_serial, card_color=None):
        for attempt in range(5):
            try:
                super().connect(card_color=card_color, card_serial=card_serial)
                break
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                else:
                    raise
        if not self.connected:
            raise ConnectionError('Error connecting to Controller.')

    def left_up(self):
        return self.sensor.leftPercent > 0
    
    def left_down(self):
        return self.sensor.leftPercent < 0

    def left_released(self):
        return self.sensor.leftPercent == 0

    def right_up(self):
        return self.sensor.rightPercent > 0

    def right_down(self):
        return self.sensor.rightPercent < 0

    def right_released(self):
        return self.sensor.rightPercent == 0
    
    def left_position(self):
        return self.sensor.leftPercent
    
    def right_position(self):
        return self.sensor.rightPercent
        
    # ── driving helper ──────────────────

    def drive(self, dm, t=100): 
        for i in range(t):
            dm.movement_move_tank(self.left_position(), self.right_position())
            time.sleep(0.1)

class colorSensor(le.ColorSensor):
    def __init__(self):
        super().__init__()

    def connect(self, card_serial, card_color=None):
        for attempt in range(5):
            try:
                super().connect(card_color=card_color, card_serial=card_serial)
                break
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                else:
                    raise
        if not self.connected:
            raise ConnectionError('Error connecting to Color Sensor.')
    
    def button_pressed(self):
        """True while the color sensor's built-in button is held down.

        `self.button` is a ButtonStateNotification kept up to date by the
        legoeducation BLE notification handler; its `.state` is
        le.BUTTON_STATE_PRESSED (1) or le.BUTTON_STATE_RELEASED (0)."""
        return self.button.state == le.BUTTON_STATE_PRESSED

    def detect_color(self):
        color_number = self.sensor.color
        color_mapping = {
            0: 'No color',
            1: 'Red',
            2: 'Yellow',
            3: 'Blue',
            4: 'Teal',
            5: 'Green',
            6: 'Purple',
            7: 'White',
            8: 'Magenta',
            9: 'Orange',
            10: 'Azure'
        }
        #detect the color, return the detected color
        return color_mapping.get(color_number, 'Unknown')

def wait(seconds: float):
    time.sleep(seconds)


azure =   "LEGO_COLOR_AZURE"
blue =   "LEGO_COLOR_BLUE"
cyan =   "LEGO_COLOR_CYAN"
green =   "LEGO_COLOR_GREEN"
red =     "LEGO_COLOR_RED"
yellow =  "LEGO_COLOR_YELLOW"
white =   "LEGO_COLOR_WHITE"
black =   "LEGO_COLOR_BLACK"
orange =  "LEGO_COLOR_ORANGE"
purple =  "LEGO_COLOR_PURPLE"
magenta = "LEGO_COLOR_MAGENTA"