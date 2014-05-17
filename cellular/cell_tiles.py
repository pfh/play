
import sys, os


#from descartes import patch


from shapely import geometry, affinity, ops

#from demakein import shape

def points_and_paths(thing):
    points = [ ]
    paths = [ ]
    point_index = { }
    def get_point(pos):
        if pos not in point_index:
            point_index[pos] = len(points)
            points.append(list(pos))
        return point_index[pos]
    
    def add_ring(ring):
        paths.append([ get_point(item) for item in ring.coords ])
    
    def add(thing):
        if type(thing) is geometry.Polygon:
            add_ring(thing.exterior)
            for item in thing.interiors:
                add_ring(item)
        elif type(thing) is geometry.MultiPolygon:
            for item in thing.geoms:
                add(item)
        else:
            assert False, repr(type(thing))

    add(thing)
    return points, paths


def as_openscad(thing):
    points, paths = points_and_paths(thing)        
    return 'polygon(points=%s,paths=%s)' % (repr(points),repr(paths))


def as_patch(thing, **kwargs):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    points, paths = points_and_paths(thing)
    
    vertices = [ ]
    codes = [ ]
    for item in paths:
        codes.extend([Path.MOVETO]+[Path.LINETO]*(len(item)-1))
        vertices.extend( points[item2] for item2 in item )
    path = Path(vertices, codes)
    return PathPatch(path, **kwargs)
        

def view(thing):
    from matplotlib import pyplot

    pyplot.gca().add_patch(as_patch(thing))    
    (minx, miny, maxx, maxy) = thing.bounds
    pyplot.xlim(minx-5,maxx+5)
    pyplot.ylim(miny-5,maxy+5)
    pyplot.gca().set_aspect(1)
    pyplot.show()


def extrude(prefix, thing):
    thing = affinity.scale(thing,2.0,2.0).buffer(-0.1)
    
    (minx, miny, maxx, maxy) = thing.bounds
    thing = affinity.translate(thing,
        -(minx+maxx)/2.0,
        -(miny+maxy)/2.0)

    with open(prefix+'.scad','wb') as f:
        f.write(
             'union() { ' +             
             'linear_extrude(height=4) '+
             as_openscad(thing.buffer(-0.5))+';'
             'translate([0,0,0.5]) ' +
             'linear_extrude(height=3.0) '+
             as_openscad(thing)+';'
             '}'
             )
                
    

directions = [(1,0),(0,1),(-1,0),(0,-1)]

def points(spec):
    x = 0
    y = 0
    d = 0
    result = [ ]
    for char in spec:
        if char == 'n':
            if not result or result[-1] != (x,y):
                result.append((x,y))
        elif char == 'l':
            d = (d+1)%4
        elif char == 'r':
            d = (d-1)%4
        elif char == 'f':
            x += directions[d][0]
            y += directions[d][1]
        else:
            assert char == ' '
    return result

def mirror(spec):
    mapper = {'l':'r','r':'l'}
    return ''.join(mapper.get(char,char) for char in spec)

def bigger(spec, scale=2):
    return spec.replace('f','f'*scale)



nub_left = 'n lffnlfn rffrffn ffrffn rfnlffn l'
nublet_left = 'n lfnlfn rffrffn ffrffn rfnlfn l'
nub_right = mirror(nub_left)
nublet_right = mirror(nublet_left)
flat = 'n ffn'
a = 'nff'
b = 'ffnl'


def cell(filled, left,mid,right):
    outline = (
        a +
        flat + 
        (nub_right if filled else flat) + 
        flat + 
        b +
        
        a + 
        flat + 
        (nublet_left if right else flat) + 
        (nublet_right if mid else flat) + 
        b +
        
        a +
        flat + 
        (nub_left if mid else flat) + 
        flat + 
        b +
        
        a + 
        (nublet_left if left else flat) +
        (nublet_right if mid else flat) +
        flat +
        b
        )
    shape = geometry.Polygon(points(outline))
    if not filled:
       shape = shape.difference(shape.buffer(-2).buffer(1))
    return shape


big_bang = geometry.Polygon(points(
    a + flat + nub_right + flat + b +
    (a + flat*3 + b) * 3
    ))

big_bang = big_bang.difference(geometry.Polygon(points(
    'ffflfffffffffr' +
    bigger(nub_right + 'll' + flat, 2)
    )))


if __name__ == '__main__':
    if not os.path.exists('output'):
       os.mkdir('output')
    
    cells = [ ]
    
    rule = int(sys.argv[1])
    
    for i in xrange(8):
        right = i&1
        mid = (i>>1)&1
        left = (i>>2)&1
        filled = (rule>>i)&1
        if not (right or mid or left or filled): continue
        cells.append(
            affinity.translate(cell(filled, left,mid,right), (i//2-2)*15, (i%2-1)*15)
            )
    
    extrude('output/big_bang', big_bang)
    extrude('output/rule-%d-cells' % rule, ops.cascaded_union(cells))
    
    
