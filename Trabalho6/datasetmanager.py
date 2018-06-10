from imutils import paths

def get_ground_truth(dataset):
    '''
    Recebe o nome do dataset a ser utilizado e retorna uma lista com os bounding boxes de ground truth
    e uma lista com as imagens referentes a esse dataset
    '''
    ####################### Dataset Professor ############################
    if dataset == "car1":
        filename = './data/Professor/car1/gtcar1.txt'
        imagesPath = './data/Professor/car1'

    elif dataset == "car2":
        filename = './data/Professor/car2/gtcar2.txt'
        imagesPath = './data/Professor/car2'
    
    ##################### Dataset Babenko ################################
    elif dataset == "Babenko_girl":
        filename = './data/Babenko/girl/gt.txt'
        imagesPath = './data/Babenko/girl'
    
    elif dataset == "Babenko_OccludedFace2":
        filename = './data/Babenko/OccludedFace2/gt.txt'
        imagesPath = './data/Babenko/OccludedFace2'
    
    elif dataset == "Babenko_surfer":
        filename = './data/Babenko/surfer/gt.txt'
        imagesPath = './data/Babenko/surfer'
    
    ################## Dataset BoBot ##########################################
    elif dataset == "BoBot_Vid_A_ball":
        filename = './data/BoBot/Vid_A_ball/gt.txt'
        imagesPath = './data/BoBot/Vid_A_ball'
    
    elif dataset == "BoBot_Vid_B_cup":
        filename = './data/BoBot/Vid_B_cup/gt.txt'
        imagesPath = './data/BoBot/Vid_B_cup'
    
    elif dataset == "BoBot_Vid_C_juice":
        filename = './data/BoBot/Vid_C_juice/gt.txt'
        imagesPath = './data/BoBot/Vid_C_juice'
    
    elif dataset == "BoBot_Vid_D_person":
        filename = './data/BoBot/Vid_D_person/gt.txt'
        imagesPath = './data/BoBot/Vid_D_person'
    
    elif dataset == "BoBot_Vid_E_person_part_occluded":
        filename = './data/BoBot/Vid_E_person_part_occluded/gt.txt'
        imagesPath = './data/BoBot/Vid_E_person_part_occluded'
    
    elif dataset == "BoBot_Vid_F_person_fully_occluded":
        filename = './data/BoBot/Vid_F_person_fully_occluded/gt.txt'
        imagesPath = './data/BoBot/Vid_F_person_fully_occluded'
    
    elif dataset == "BoBot_Vid_G_rubikscube":
        filename = './data/BoBot/Vid_G_rubikscube/gt.txt'
        imagesPath = './data/BoBot/Vid_G_rubikscube'
    
    elif dataset == "BoBot_Vid_H_panda":
        filename = './data/BoBot/Vid_H_panda/gt.txt'
        imagesPath = './data/BoBot/Vid_H_panda'
    
    elif dataset == "BoBot_Vid_I_person_crossing":
        filename = './data/BoBot/Vid_I_person_crossing/gt.txt'
        imagesPath = './data/BoBot/Vid_I_person_crossing'
    
    elif dataset == "BoBot_Vid_J_person_floor":
        filename = './data/BoBot/Vid_J_person_floor/gt.txt'
        imagesPath = './data/BoBot/Vid_J_person_floor'
    
    elif dataset == "BoBot_Vid_K_cup":
        filename = './data/BoBot/Vid_K_cup/gt.txt'
        imagesPath = './data/BoBot/Vid_K_cup'
    
    elif dataset == "BoBot_Vid_L_coffee":
        filename = './data/BoBot/Vid_L_coffee/gt.txt'
        imagesPath = './data/BoBot/Vid_L_coffee'
    
    ############################ Dataset Cehovin ###########################
    elif dataset == "Cehovin_dinosaur":
        filename = './data/Cehovin/dinosaur/gt.txt'
        imagesPath = './data/Cehovin/dinosaur'
    
    elif dataset == "Cehovin_gymnastics":
        filename = './data/Cehovin/gymnastics/gt.txt'
        imagesPath = './data/Cehovin/gymnastics'
    
    elif dataset == "Cehovin_hand":
        filename = './data/Cehovin/hand/gt.txt'
        imagesPath = './data/Cehovin/hand'
    
    elif dataset == "Cehovin_hand2":
        filename = './data/Cehovin/hand2/gt.txt'
        imagesPath = './data/Cehovin/hand2'
    
    elif dataset == "Cehovin_torus":
        filename = './data/Cehovin/torus/gt.txt'
        imagesPath = './data/Cehovin/torus'
    
    ######################## Dataset Ellis_ijcv2011 ###################################
    elif dataset == "Ellis_ijcv2011_head_motion":
        filename = './data/Ellis_ijcv2011/head_motion/gt.txt'
        imagesPath = './data/Ellis_ijcv2011/head_motion'
    
    elif dataset == "Ellis_ijcv2011_track_running":
        filename = './data/Ellis_ijcv2011/track_running/gt.txt'
        imagesPath = './data/Ellis_ijcv2011/track_running'
    
    elif dataset == "Ellis_ijcv2011_shaking_camera":
        filename = './data/Ellis_ijcv2011/shaking_camera/gt.txt'
        imagesPath = './data/Ellis_ijcv2011/shaking_camera'
    
    ######################## Dataset Godec ############################################
    elif dataset == "Godec_cliff-dive1":
        filename = './data/Godec/cliff-dive1/gt.txt'
        imagesPath = './data/Godec/cliff-dive1'
    
    elif dataset == "Godec_cliff-dive2":
        filename = './data/Godec/cliff-dive2/gt.txt'
        imagesPath = './data/Godec/cliff-dive2'
    
    elif dataset == "Godec_motocross1":
        filename = './data/Godec/motocross1/gt.txt'
        imagesPath = './data/Godec/motocross1'
    
    elif dataset == "Godec_motocross2":
        filename = './data/Godec/motocross2/gt.txt'
        imagesPath = './data/Godec/motocross2'
    
    elif dataset == "Godec_mountain-bike":
        filename = './data/Godec/mountain-bike/gt.txt'
        imagesPath = './data/Godec/mountain-bike'
    
    elif dataset == "Godec_skiing":
        filename = './data/Godec/skiing/gt.txt'
        imagesPath = './data/Godec/skiing'
    
    elif dataset == "Godec_volleyball":
        filename = './data/Godec/volleyball/gt.txt'
        imagesPath = './data/Godec/volleyball'
    
    ################### Dataset Kalal #########################################
    elif dataset == "Kalal_car":
        filename = './data/Kalal/car/gt.txt'
        imagesPath = './data/Kalal/car'

    elif dataset == "Kalal_CarChase":
        filename = './data/Kalal/CarChase/gt.txt'
        imagesPath = './data/Kalal/CarChase'

    elif dataset == "Kalal_david":
        filename = './data/Kalal/david/gt.txt'
        imagesPath = './data/Kalal/david'

    elif dataset == "Kalal_jumping":
        filename = './data/Kalal/jumping/gt.txt'
        imagesPath = './data/Kalal/jumping'

    elif dataset == "Kalal_Motocross":
        filename = './data/Kalal/Motocross/gt.txt'
        imagesPath = './data/Kalal/Motocross'

    elif dataset == "Kalal_Panda":
        filename = './data/Kalal/Panda/gt.txt'
        imagesPath = './data/Kalal/Panda'

    elif dataset == "Kalal_pedestrian3":
        filename = './data/Kalal/pedestrian3/gt.txt'
        imagesPath = './data/Kalal/pedestrian3'

    elif dataset == "Kalal_pedestrian4":
        filename = './data/Kalal/pedestrian4/gt.txt'
        imagesPath = './data/Kalal/pedestrian4'

    elif dataset == "Kalal_pedestrian5":
        filename = './data/Kalal/pedestrian5/gt.txt'
        imagesPath = './data/Kalal/pedestrian5'

    elif dataset == "Kalal_Volkswagen":
        filename = './data/Kalal/Volkswagen/gt.txt'
        imagesPath = './data/Kalal/Volkswagen'
    
    ################## Dataset Kwon ############################
    elif dataset == "Kwon_diving":
        filename = './data/Kwon/diving/gt.txt'
        imagesPath = './data/Kwon/diving'
    
    elif dataset == "Kwon_gym":
        filename = './data/Kwon/gym/gt.txt'
        imagesPath = './data/Kwon/gym'
    
    elif dataset == "Kwon_jump":
        filename = './data/Kwon/jump/gt.txt'
        imagesPath = './data/Kwon/jump'
    
    elif dataset == "Kwon_trans":
        filename = './data/Kwon/trans/gt.txt'
        imagesPath = './data/Kwon/trans'
    
    ################## Dataset Kwon_VTD ########################
    elif dataset == "Kwon_VTD_animal":
        filename = './data/Kwon_VTD/animal/gt.txt'
        imagesPath = './data/Kwon_VTD/animal'
    
    elif dataset == "Kwon_VTD_basketball":
        filename = './data/Kwon_VTD/basketball/gt.txt'
        imagesPath = './data/Kwon_VTD/basketball'
    
    elif dataset == "Kwon_VTD_football":
        filename = './data/Kwon_VTD/football/gt.txt'
        imagesPath = './data/Kwon_VTD/football'
    
    elif dataset == "Kwon_VTD_shaking":
        filename = './data/Kwon_VTD/shaking/gt.txt'
        imagesPath = './data/Kwon_VTD/shaking'
    
    elif dataset == "Kwon_VTD_singer1":
        filename = './data/Kwon_VTD/singer1/gt.txt'
        imagesPath = './data/Kwon_VTD/singer1'
    
    elif dataset == "Kwon_VTD_singer1(lowfps)":
        filename = './data/Kwon_VTD/singer1(lowfps)/gt.txt'
        imagesPath = './data/Kwon_VTD/singer1(lowfps)'

    elif dataset == "Kwon_VTD_singer2":
        filename = './data/Kwon_VTD/singer2/gt.txt'
        imagesPath = './data/Kwon_VTD/singer2'

    elif dataset == "Kwon_VTD_skating1(lowfps)":
        filename = './data/Kwon_VTD/skating1(lowfps)/gt.txt'
        imagesPath = './data/Kwon_VTD/skating1(lowfps)'

    elif dataset == "Kwon_VTD_skating2":
        filename = './data/Kwon_VTD/skating2/gt.txt'
        imagesPath = './data/Kwon_VTD/skating2'

    ############### Dataset Other #####################
    elif dataset == "Other_Asada":
        filename = './data/Other/Asada/gt.txt'
        imagesPath = './data/Other/Asada'
    
    elif dataset == "Other_drunk2":
        filename = './data/Other/drunk2/gt.txt'
        imagesPath = './data/Other/drunk2'
    
    elif dataset == "Other_dudek-face":
        filename = './data/Other/dudek-face/gt.txt'
        imagesPath = './data/Other/dudek-face'
    
    elif dataset == "Other_faceocc1":
        filename = './data/Other/faceocc1/gt.txt'
        imagesPath = './data/Other/faceocc1'
    
    elif dataset == "Other_figure_skating":
        filename = './data/Other/figure_skating/gt.txt'
        imagesPath = './data/Other/figure_skating'
    
    elif dataset == "Other_woman":
        filename = './data/Other/woman/gt.txt'
        imagesPath = './data/Other/woman'
    
    #################### Dataset Prost #####################
    elif dataset == "PROST_board":
        filename = './data/PROST/board/gt.txt'
        imagesPath = './data/PROST/board'
    
    elif dataset == "PROST_box":
        filename = './data/PROST/box/gt.txt'
        imagesPath = './data/PROST/box'
    
    elif dataset == "PROST_lemming":
        filename = './data/PROST/lemming/gt.txt'
        imagesPath = './data/PROST/lemming'
    
    elif dataset == "PROST_liquor":
        filename = './data/PROST/liquor/gt.txt'
        imagesPath = './data/PROST/liquor'
    
    ################## Dataset Ross #####################
    elif dataset == "Ross_car11":
        filename = './data/Ross/car11/gt.txt'
        imagesPath = './data/Ross/car11'
    
    elif dataset == "Ross_dog1":
        filename = './data/Ross/dog1/gt.txt'
        imagesPath = './data/Ross/dog1'
    
    elif dataset == "Ross_Sylvestr":
        filename = './data/Ross/Sylvestr/gt.txt'
        imagesPath = './data/Ross/Sylvestr'
    
    elif dataset == "Ross_trellis":
        filename = './data/Ross/trellis/gt.txt'
        imagesPath = './data/Ross/trellis'
    
    ################## Dataset Thang #####################
    elif dataset == "Thang_coke":
        filename = './data/Thang/coke/gt.txt'
        imagesPath = './data/Thang/coke'
    
    elif dataset == "Thang_person":
        filename = './data/Thang/person/gt.txt'
        imagesPath = './data/Thang/person'
    
    elif dataset == "Thang_tiger1":
        filename = './data/Thang/tiger1/gt.txt'
        imagesPath = './data/Thang/tiger1'
    
    elif dataset == "Thang_tiger2":
        filename = './data/Thang/tiger2/gt.txt'
        imagesPath = './data/Thang/tiger2'
    
    ################# Dataset Wang #########################
    elif dataset == "Wang_bird_1":
        filename = './data/Wang/bird_1/gt.txt'
        imagesPath = './data/Wang/bird_1'
    
    elif dataset == "Wang_bird_2":
        filename = './data/Wang/bird_2/gt.txt'
        imagesPath = './data/Wang/bird_2'
    
    elif dataset == "Wang_bolt":
        filename = './data/Wang/bolt/gt.txt'
        imagesPath = './data/Wang/bolt'
    
    elif dataset == "Wang_girl_mov":
        filename = './data/Wang/girl_mov/gt.txt'
        imagesPath = './data/Wang/girl_mov'
    
    else:
        raise NameError ("There is no {} dataset".format(dataset))

    #Create a list to store the ground truth bounding boxes
    bounding_boxes = []

    with open(filename) as f:
        data = f.readlines()

    print("[DEBUG] dataset split = ", dataset.split("_"))

    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        if (dataset.split("_")[0] + '_' + dataset.split("_")[1]) == "Kwon_VTD":
            #print("[DEBUG] teste = ", data[i].split(" "))
            top, left, bottom, right = data[i].split(" ")
        else:
            top, left, bottom, right = data[i].split(",")
        bb = [float(top), float(left), float(bottom), float(right)]
        bounding_boxes.append(bb)

    imagesList = sorted(list(paths.list_images(imagesPath)))
    #print(imagesList)

    return bounding_boxes, imagesList