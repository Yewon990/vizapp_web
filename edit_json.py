import json

class Edit_Json():
    # def __init__(self, video_segment_list):
    #     self.video_segment_list = video_segment_list
    
    def get_seg_num(self, json_data, click_select):
        if json_data == None:
            # return self.video_segment_list
            return

        elif click_select == "click":
            points = json_data['points'][0]
            seg_num = points['customdata'][0]
            #self.video_segment_list.append(seg_num)
            return int(seg_num)
        
        elif click_select == "select":
            segment_list = []
            points = json_data['points']
            
            for point in points:
                seg_num = point['customdata'][0]
                segment_list.append(seg_num)
                # self.video_segment_list.append(seg_num)
            
            return list(segment_list)