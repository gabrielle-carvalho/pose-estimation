from enum import Enum, auto
import time
from mission_control.skills.recognize_object import RecognizeObject
from mission_control.missions.mission import Mission

from mission_control.camera3dconverter.camera_3d_converter import Camera3DConverter
from mission_control.objects import ObjectBoundingBox 

import threading
import time
from mission_control.generate_pdf.save_pdf import SavePDF

from mission_control.skills_manager import Ski



class NavigationFollowMe(Mission):
    class Steps(Enum):
        READY_TO_START_TASK = auto()
        # ENTERING_THE_ARENA = auto()
        # GOING_TO_FIRST_WAYPOINT = auto()
        # GOING_TO_SECOND_WAYPOINT = auto()
        WAIT_FOR_OPERATOR = auto()
        TRAINING_OPERATOR = auto()
        REGISTER_FACE = auto()
        REGISTER_FINISHED = auto()
        RECOGNIZE_PERSON = auto()
        RECOGNIZE_PERSON_POSITION = auto()
        CALL_OPERATOR = auto()
        FOLLOW_OPERATOR = auto()
        CHECK_POSE = auto()
        GOING_BACK = auto()
        LEAVING_THE_ARENA = auto()
        END = auto()

    @property
    def required_services(self) -> list[str]:
        return ['speak', 'go_to_location', 'RecognizeObject', 'RecognizeObjectPosition', 'recognize_faces', 'register_face'] #people

    def __init__(self, logger, node):
        super().__init__(logger=logger, node=node)
        self.current_step = self.Steps.READY_TO_START_TASK
        self.mission_completed = False
        self.convert_3d = Camera3DConverter()

        self.count = 0


        self.wait_start_time = None
        self.detected_objects = []
        self.objects_positions = []
        self.object_to_point = ObjectBoundingBox()
        self.object_position = None
        self.pdf_saver = SavePDF(mission_name="object_recognition")

    def execute(self):
        if self.mission_completed:
            return

        match(self.current_step):
            case self.Steps.READY_TO_START_TASK:
                if self.skills_manager.execute('Speak', text='Please, open the door for me.'):
                    self.current_step = self.Steps.RECOGNIZE_PERSON

            # case self.Steps.ENTERING_THE_ARENA:
            #     time.sleep(15)
            #     if self.skills_manager.execute('GoToLocation', location_name='front_door'):
            #         self.current_step = self.Steps.GOING_TO_FIRST_WAYPOINT

            # case self.Steps.GOING_TO_FIRST_WAYPOINT:
            #     if self.skills_manager.execute('Speak', text='I just enter the arena, now i am going to the first waypoint.'):
            #         if self.skills_manager.execute('GoToLocation', location_name='waypoint1'):
            #             self.skills_manager.execute('Speak', text='Arrived at the first waypoint')
            #             self.current_step = self.Steps.GOING_TO_SECOND_WAYPOINT
            
            # case self.GOING_TO_SECOND_WAYPOINT:
            #     if self.skills_manager.execute('Speak', text='I am going to the second waypoint.'):
            #         if self.skills_manager.execute('GoToLocation', location_name='waypoint2'):
            #             self.skills_manager.execute('Speak', text='Arrived at the second waypoint')
            #             self.current_step = self.Steps.WAIT_FOR_OPERATOR
            
            case self.Steps.WAIT_FOR_OPERATOR:
                if self.skills_manager.execute('Speak', text='I am wait for the operator to step in front of me.'):
                    time.sleep(5)
                    self.current_step = self.Steps.TRAINING_OPERATOR

            case self.Steps.TRAINING_OPERATOR:
                if self.skills_manager.execute('Speak', text='Please stand still while I learn to recognize you.'):
                    self.current_step = self.Steps.REGISTER_FACE

            case self.Steps.REGISTER_FACE:
                if self.skills_manager.execute('RegisterFace', name='operator'):
                    self.current_step = self.Steps.REGISTER_FINISHED
            
            case self.Steps.REGISTER_FINISHED: #CONGELAR BBOX PARA ACOMPANHAR
                if self.skills_manager.execute('Speak', text='I will start to follow you.'):
                    time.sleep(6)
                    self.current_step = self.Steps.RECOGNIZE_PERSON

            case self.Steps.RECOGNIZE_PERSON:
                has_recognized = self.skills_manager.execute('RecognizeObject')
                if has_recognized:
                    self.detected_objects = (
                        self.skills_manager
                        .get_skill_object('RecognizeObject')
                        .get_recognized_objects()
                    )
                    if self.detected_objects:
                        for obj in self.detected_objects:
                            self.pdf_saver.add_object(
                                name=obj.name,
                                image_path=getattr(obj, "image_path", None)
                            )
                        self.current_step = self.Steps.RECOGNIZE_PERSON_POSITION
                    else:
                        self.current_step = self.Steps.CALL_OPERATOR

            case self.Steps.RECOGNIZE_PERSON_POSITION:
                persons = [obj for obj in self.detected_objects if obj.name == 'person']

                if not persons:
                    self.get_logger().warn("none person have been found, back to the recognition step")
                    self.current_step = self.Steps.CALL_OPERATOR
                    return

                screen_center_x, screen_center_y = (640 / 2, 480 / 2) # VERIFICAR VALORES DA ASUS!
                
                def get_distance_from_center(person_obj):
                    center_x = (person_obj.x1 + person_obj.x2) / 2
                    center_y = (person_obj.y1 + person_obj.y2) / 2
                    return ((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)**0.5

                closest_person = min(persons, key=get_distance_from_center)
                
                self.get_logger().info(f"person more closer of middle. coordinates: ({closest_person.x1}, {closest_person.y1})")
                
                closest_bbox = (
                    int(closest_person.x1),
                    int(closest_person.y1),
                    int(closest_person.x2),
                    int(closest_person.y2)
                )
                
                has_recognized_position = self.skills_manager.execute('RecognizeObjectPosition', bboxes=[closest_bbox])
                
                if has_recognized_position:
                    self.object_position = (
                        self.skills_manager
                        .get_skill_object('RecognizeObjectPosition')
                        .get_recognized_object_positions()
                    )
                    if self.object_position:
                        self.get_logger().info(f"position: {self.object_position}")
                        self.current_step = self.Steps.FOLLOW_OPERATOR
                    else:
                        self.get_logger().warn("Não foi possível obter a posição 3D da pessoa")
                        self.current_step = self.Steps.FOLLOW_OPERATOR 
                else:
                    pass

            case self.Steps.CALL_OPERATOR:
                if self.skills_manager.execute('Speak', text='I can not see you'):
                    self.current_step = self.Steps.RECOGNIZE_PERSON

            case self.Steps.FOLLOW_OPERATOR:                                                                                                                        
                if self.skills_manager.execute('Speak', text='i am following you'):
                    time.sleep(3)
                    #IMPLEMENTAR
                    self.count += 1 
                    self.get_logger().info(f"Iniciando checagem de pose. Tentativa: {self.count}")
                    self.current_step = self.Steps.CHECK_POSE
            
            case self.Steps.CHECK_POSE:
                if self.count >= 3:
                    if self.skills_manager.execute('Speak', text='Ok, pose recognized'):
                        self.count = 0 
                        self.current_step = self.Steps.GOING_BACK_SPEAK 
                else:
                    if self.skills_manager.execute('Speak', text='I do not recognize pose to stop'):
                        self.current_step = self.Steps.FOLLOW_OPERATOR

            case self.GOING_BACK:
                if self.skills_manager.execute('Speak', text='I am going back to the second waypoint.'):
                    if self.skills_manager.execute('GoToLocation', location_name='waypoint2'):
                        self.skills_manager.execute('Speak', text='Arrived at the second waypoint')
                        self.current_step = self.Steps.LEAVING_THE_ARENA

            case self.Steps.LEAVING_THE_ARENA:
                if self.skills_manager.execute('Speak', text='I am now leaving the arena.'):
                    if self.skills_manager.execute('GoToLocation', location_name='back_door'):
                        self.current_step = self.Steps.END
            
            case self.Steps.END:
                self.skills_manager.execute('Speak', text='Follow me mission finished.')
                self.mission_completed = True

