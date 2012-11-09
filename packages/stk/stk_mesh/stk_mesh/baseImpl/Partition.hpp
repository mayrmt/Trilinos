/*
 * Partition.hpp
 *
 */

#ifndef STK_MESH_IMPL_PARTITION_HPP_
#define STK_MESH_IMPL_PARTITION_HPP_

#include <stk_mesh/base/Types.hpp>


namespace stk {
namespace mesh {

namespace impl {

class BucketRepository;

class Partition
{
    friend class BucketRepository;
public:
    Partition(BucketRepository *repo, EntityRank rank);

    virtual ~Partition();

    std::ostream &streamit(std::ostream &os) const;

    const EntityRank get_rank() const { return m_rank; }

    inline std::vector<Bucket *>::iterator begin();
    inline std::vector<Bucket *>::iterator end();

    inline std::vector<Bucket *>::const_iterator begin() const;
    inline std::vector<Bucket *>::const_iterator end() const;

    inline bool empty() const;

    const std::vector<PartOrdinal> &get_legacy_partition_id() const
    {
        return m_extPartitionKey;
    }

    bool add(Entity entity);

    void move_to(Entity entity, Partition &dst_partition);

    bool remove(Entity entity);

    /// Compress this partion into a single bucket of sorted Entities.
    void compress();

    /// Sort the entities in this partition by EntityKey.
    void sort();
    
    inline bool belongs(Bucket *bkt) const;

    size_t compute_size()
    {
        size_t partition_size = 0;
        std::vector<Bucket *>::iterator buckets_end = end();
        for (std::vector<Bucket *>::iterator b_i = begin(); b_i != buckets_end; ++b_i)
        {
            partition_size += (*b_i)->size();
        }
        return partition_size;
    }

    bool modify_bucket_set() const { return m_modifyBucketSet; }

    size_t num_buckets() const
    {
        return (m_modifyBucketSet ? m_buckets.size() : m_endBucketIndex - m_beginBucketIndex);
    }

    // Just for unit testing.  Remove after refactor.
    static BucketRepository &getRepository(stk::mesh::BulkData &mesh);

    // Just for unit testing.  DOES NOT SYNC DATA.
    void reverseEntityOrderWithinBuckets();

private:

    BucketRepository *m_repository;
    EntityRank m_rank;

    std::vector<PartOrdinal> m_extPartitionKey;

    std::vector<Bucket *> m_buckets;

    unsigned m_beginBucketIndex;
    unsigned m_endBucketIndex;

    // Flag that the set of buckets, and not just their contents, is being modified.
    bool m_modifyBucketSet;

    bool no_buckets() const;

    // Take the buckets from the repository.
    bool take_bucket_control();

    Bucket *get_bucket_for_adds();

};


std::ostream &operator<<(std::ostream &, const stk::mesh::impl::Partition &);


} // impl
} // mesh
} // stk



#endif /* BUCKETFAMILY_HPP_ */
