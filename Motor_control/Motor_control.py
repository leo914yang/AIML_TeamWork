import time
import RPi.GPIO as GPIO


def angle(a):
    return round(float(a) / float(360), 2)


def Direction(steps, STEPS_PER_REVOLUTION):
    if STEPS_PER_REVOLUTION > 0:
        return 1
    elif STEPS_PER_REVOLUTION < 0:
        return -1


def Run(angle, direction=1, wait_time=5/float(1000), STEPPER_PINS=[17,18,27,22]):
    GPIO.setmode(GPIO.BCM)
    
    STEPS_PER_REVOLUTION = int(64 * 64 * angle)
    SEQUENCE = [[1,0,0,0],
                [1,1,0,0],
                [0,1,0,0],
                [0,1,1,0],
                [0,0,1,0],
                [0,0,1,1],
                [0,0,0,1],
                [1,0,0,1]]
    
    for pin in STEPPER_PINS:
        GPIO.setup(pin,GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    SEQUENCE_COUNT = len(SEQUENCE)
    PINS_COUNT = len(STEPPER_PINS)
    
    sequence_index = 0 # 記錄目前進行到哪個步驟的變數
    steps = 0 # 記錄目前已移動步數
    
    try:
        print('開始程式')
        while True:
            if steps == STEPS_PER_REVOLUTION:
                break
            for pin in range(PINS_COUNT): # 給3(?)根pin低電位
                GPIO.output(STEPPER_PINS[pin], SEQUENCE[sequence_index][pin])
    
            steps += direction

            direction = Direction(steps, STEPS_PER_REVOLUTION)
    
            sequence_index += direction
            sequence_index %= SEQUENCE_COUNT

            time.sleep(wait_time)
    except KeyboardInterrupt:
        print('關閉程式')
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    Run(float(angle(angle := input('Input angle: '))))
