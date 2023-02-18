from face_anonymizing import Driver

if __name__ == '__main__':
    foo = Driver(i='test_data/', o='test_results/', device='cpu')  # Use 'cuda:0' instead of 'cpu' if CUDA is available
    s = len(foo.img_paths)
    for i in range(s):
        img = foo.get_next_img()
        if isinstance(img, str):
            continue
        bboxes = foo.detect_faces(img)
        # img_with_boxes = foo.merge_faces_with_bboxes(img, bboxes)
        blured_img = foo.blur_img(img, bboxes)
        foo.save_img(blured_img)
