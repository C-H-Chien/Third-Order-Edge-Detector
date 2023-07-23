#ifndef CURVELETMAP_HPP
#define CURVELETMAP_HPP

/*****************************************************************************
 // file: curveletmap.h
 // brief: Curvelet Map data structure
 // author: Xiaoyan Li
 // date: 01/26/2016
 ******************************************************************************/
#include "curveletmap.hpp"
#include <set>
#include "edgemap.hpp"
#include "curvelet.hpp"

//: This class stores the map of curvelets formed by the SEL edge linker.
//  This is an intermediate structure before actual edge linking occurs.
class curveletmap
{
public:
    
    //: constructor
    curveletmap(edgemap* EM=0);
    
    //: destructor
    ~curveletmap();
    
    //: to check if the Curvelet map is valid
    bool is_valid() { return _map.size()>0 && _map.size()==_EM->num_edgels(); };
    
    //: set the edgemap
    //void set_edgemap(edgemap* EM) { _EM = EM; resize(EM->num_edgels()); }
    
    //: access the curvelets for an edge using its id
    const cvlet_list& getcurvelets(unsigned id) const { return _map[id]; };
    cvlet_list& getcurvelets(unsigned id) { return _map[id]; };
    
    //: add a curvelet to an edgel
    void add_curvelet(curvelet* curvelet);
    
    //: remove a curvelet from this edgel
    // void remove_curvelet(dbdet_curvelet* curvelet);
    
    //: delete all the curvelets formed by this edgel
    // void delete_all_curvelets(dbdet_edgel* e);
    
    //: does this curvelet exist at this edgel?
    curvelet* does_curvelet_exist(edgel* e, std::deque<edgel*> & chain);
    
    //: return the number of curvelets in this curvelet map
    unsigned num_curvelets();
    
    //: record the curvelet map data in array format
    void get_output_array(arrayi &id_chain, arrayd &curvelet_info);
    
    // forms a curvelet map which is a mapping from each edgel to all curvelets it participates in
    void form_full_cvlet_map();
    
    //: print info to file
    void print(std::ostream&);
    
private:
    
    //: The edgemap on which these curvelets have been formed
    //  (due to this smart pointer to the edgemap, the curvelets remain valid even if the edgemap is deleted elsewhere)
    edgemap* _EM;
    
    //: The curvelet map, indexed by edgel IDs
    std::vector<cvlet_list > _map;
    
    bool _fullcvmap;
    
    //: resize the graph
    void resize(unsigned size);
    
    //: clear the graph
    void clear();//!!!!!!!!
    
    friend class edgemap;
};



//: constructor
curveletmap::curveletmap(edgemap* EM):
  _EM(EM), _fullcvmap(false)
{
    resize(EM->num_edgels());
}

//: destructor
curveletmap::~curveletmap()
{
  clear(); //delete everything upon exit
  _EM=0; //delete the reference to the edgemap
}

//: resize the graph
void curveletmap::resize(unsigned size)
{ 
  if (size!=_map.size())
    clear();

  _map.resize(size);
}

//: clear the graph
void curveletmap::clear()
{
  std::set<curvelet*> deleted;
  //delete all the curvelets in the map
  for (unsigned i=0; i<_map.size(); i++)
  {
    //delete all the curvelets formed by this edgel
    cvlet_list::iterator p_it;
    for (p_it = _map[i].begin(); p_it != _map[i].end(); p_it++) {
        if (deleted.find(*p_it) == deleted.end()) {
            delete (*p_it);
            deleted.insert(*p_it);
        }
    }
    _map[i].clear();
  }
  _map.clear();
  _fullcvmap = false;
}

//: add a curvelet to this edgel
void curveletmap::add_curvelet(curvelet* curvelet)
{
  _map[curvelet->_ref_edgel->_id].push_back(curvelet);
}

//: does the current curvelet exist?
curvelet* curveletmap::does_curvelet_exist(edgel* e, std::deque<edgel*> & chain)
{
  // if(_map.empty()) return 0;
  //go over all the curvelets of the current size formed by the current edgel
  cvlet_list::iterator cv_it;
  for ( cv_it = _map[e->_id].begin(); cv_it != _map[e->_id].end(); cv_it++){
    curvelet* cvlet = (*cv_it);

    if (cvlet->_edgel_chain.size() != chain.size())
      continue;

    bool cvlet_exists = true; //reset flag
    for (unsigned k=0; k<chain.size(); k++)
      cvlet_exists = cvlet_exists && (cvlet->_edgel_chain[k]==chain[k]);

    //the flag will remain true only if all the edgels match
    if (cvlet_exists)
      return cvlet; //return matching curvelet
  }

  return 0; //curvelet does not exist
}

void curveletmap::print(std::ostream& os)
{
    for (unsigned i = 0; i<_map.size(); i++) {
        for (cvlet_list::iterator it=_map[i].begin(); it!=_map[i].end(); it++) {
            (*it)->print(os);
        }
    }
}

unsigned curveletmap::num_curvelets()
{
    unsigned count = 0;
    for (unsigned i = 0; i<_map.size(); i++) {
        count += _map[i].size();
    }
    return count;
}

void curveletmap::get_output_array(arrayi &id_chain, arrayd &curvelet_info)
{
    unsigned count = 0;
    for (unsigned i = 0; i<_map.size(); i++) {
        for (cvlet_list::iterator it=_map[i].begin(); it!=_map[i].end(); it++) {
            // anchor edgel, convert to matlab index
            id_chain.set_val(count, 0, int(i+1));
            (*it)->set_output(id_chain, curvelet_info, count, 1);
            count++;
        }
    }
    
}

void curveletmap::form_full_cvlet_map()
{
    // This method forms a curvelet map which is a mapping from each edgel to all the curvelets it participates in and not just the ones anchored to it.
    if (_fullcvmap) return;
    
    for (unsigned i=0; i<_EM->_list.size(); i++){
        edgel* eA = _EM->_list[i];
        
        //add all the curvelets anchored at this edgel to all the other edgels in it
        cvlet_list::iterator cv_it = getcurvelets(i).begin();
        for ( ; cv_it!=getcurvelets(i).end(); cv_it++){
            curvelet* cvlet = (*cv_it);
            
            //only add the ones that are anchored to this edgel
            if (cvlet->_ref_edgel != eA)
                continue;
            
            //add this curvelet to each of the edgels of the grouping
            for (unsigned n=0; n<cvlet->_edgel_chain.size(); n++){
                edgel* eB = cvlet->_edgel_chain[n];
                
                //make sure that there are no duplicates (this is not strictly necessary)
                bool cvlet_exists = false; //reset flag
                
                //go over all the curvelets (not just anchored) formed by the current edgel
                cvlet_list::iterator cv_it2 = getcurvelets(eB->_id).begin();
                for ( ; cv_it2!=getcurvelets(eB->_id).end(); cv_it2++){
                    curvelet* cvlet2 = (*cv_it2);
                    
                    if (cvlet2==cvlet){
                        cvlet_exists=true;
                        break;
                    }
                    
                    if (cvlet2->_edgel_chain.size() != cvlet->_edgel_chain.size())
                        continue;
                    
                    bool exists = true;
                    for (unsigned k=0; k<cvlet2->_edgel_chain.size(); k++)
                        exists = exists && (cvlet2->_edgel_chain[k]==cvlet->_edgel_chain[k]);
                    
                    //the flag will remain true only if all the edgels match
                    if (exists){
                        cvlet_exists= true;
                        break;
                    }
                }
                
                if (!cvlet_exists)
                    getcurvelets(eB->_id).push_back((*cv_it)); //insert this into the map
            }
        }
    }
    _fullcvmap = true;
}



#endif // CURVELETMAP_HPP