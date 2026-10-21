"""Browser regression for painting, transparent background, undo, save and reload.

Requires the optional playwright package and its Chromium browser.
Uses intercepted HTTP fixtures; never writes actual annotation masks.
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    html = Path(__file__).with_name('spill_editor.html').read_text()
    buffer = io.BytesIO()
    Image.new('RGB', (256, 256), (80, 80, 80)).save(buffer, 'PNG')
    frame = buffer.getvalue()
    saved = {}
    rows = [dict(video_id='fixture', slot=i, source_frame=i, file=f'{i}.png',
                 url=f'/frame/{i}', mask_url=f'/mask/{i}', annotated=False) for i in range(2)]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1200, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def route(request):
            url = request.request.url.split('http://spill.test')[1].split('?')[0]
            if url == '/': request.fulfill(body=html, content_type='text/html')
            elif url == '/api/items': request.fulfill(json=rows)
            elif url.startswith('/frame/'): request.fulfill(body=frame, content_type='image/png')
            elif url.startswith('/mask/'): request.fulfill(body=saved[url.rsplit('/', 1)[1]+'.png'], content_type='image/png')
            elif url == '/api/save':
                data = request.request.post_data_json
                saved[data['file']] = base64.b64decode(data['data'].split(',')[1])
                rows[int(data['file'].split('.')[0])]['annotated'] = True
                request.fulfill(json={'saved': True})
            else: request.fulfill(status=404)

        page.route('http://spill.test/**', route)
        page.goto('http://spill.test/')
        page.wait_for_function('!busy')
        pixel = lambda x,y: page.evaluate(f'Array.from(ctx.getImageData({x},{y},1,1).data)')
        assert pixel(10,10) == [80,80,80,255], 'Empty mask must leave frame unchanged'
        box = page.locator('#canvas').bounding_box()
        def position(x,y): return box['x']+x/256*box['width'], box['y']+y/256*box['height']
        page.mouse.move(*position(60,100)); page.mouse.down()
        page.mouse.move(*position(160,100), steps=20); page.mouse.up()
        assert page.evaluate('mc.getImageData(100,100,1,1).data[0]') == 255
        assert pixel(10,10) == [80,80,80,255], 'Painting must not tint background'
        assert pixel(100,100)[0] > 80
        page.click('#undo')
        assert page.evaluate('mc.getImageData(100,100,1,1).data[0]') == 0
        page.click('#redo'); page.click('#erase'); page.mouse.click(*position(100,100))
        assert page.evaluate('mc.getImageData(100,100,1,1).data[0]') == 0
        page.click('#undo')
        page.click('#save'); page.wait_for_function('index===1 && !busy')
        assert set(Image.open(io.BytesIO(saved['0.png'])).getdata()) == {(0,0,0,255),(255,255,255,255)}
        page.click('#previous'); page.wait_for_function('index===0 && !busy')
        assert page.evaluate('mc.getImageData(100,100,1,1).data[0]') == 255
        assert pixel(10,10) == [80,80,80,255]
        page.screenshot(path='/tmp/spill_brush_editor.png', full_page=True)
        assert not errors, errors
        browser.close()
    print(json.dumps({'brush_drag': 'passed', 'background_transparency': 'passed',
                      'eraser_undo_redo': 'passed', 'binary_save_reload': 'passed'}))


if __name__ == '__main__': main()
