$pp = 'D:\Final Year Project\deliverables_v2\presentation\Deepfake_Detection_Presentation_v2.pptx'
$out = 'D:\Final Year Project\deliverables_v2\presentation\_render'
$a = New-Object -ComObject PowerPoint.Application
$p = $a.Presentations.Open($pp, -1, 0, 0)
for ($i=1; $i -le $p.Slides.Count; $i++) { $p.Slides.Item($i).Export("$out\s{0:d2}.png" -f $i, 'PNG', 1280, 720) }
$p.Close(); $a.Quit()
